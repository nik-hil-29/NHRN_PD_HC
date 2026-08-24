"""
model.py
========
Neuromorphic Hierarchical Resonance Network (NHRN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _valid_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Largest divisor of num_channels that is <= max_groups, for GroupNorm.
    GroupNorm normalizes per-sample (unlike BatchNorm1d, which normalizes
    per-batch and freezes running statistics for eval) — this removes the
    train/eval statistic mismatch that BatchNorm1d can produce with small
    subject counts and highly correlated (50%-overlapping) training windows,
    a likely contributor to the near-chance validation accuracy observed
    even on single-site (UCSD-only) training.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class _GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -lambda_ in
    the backward pass. Standard building block for domain-adversarial
    training (Ganin & Lempitsky, 2015)."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps _GradientReversalFunction with a settable strength ``lambda_``."""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)


class SiteDiscriminator(nn.Module):
    """
    Small auxiliary head that tries to predict recording SITE (not PD/HC)
    from the same pooled feature the main classifier uses. Attached via a
    GradientReversalLayer: the discriminator itself is trained normally to
    predict site as well as it can, but its gradient is reversed before
    reaching the shared backbone — so the backbone is pushed to produce
    features the site classifier CANNOT use, i.e. site-invariant features.

    This directly targets a specific, distinct failure mode from ordinary
    overfitting: a network with genuinely low overfitting-surface (small
    dropout, no memorization-prone components) can still plateau at
    mediocre validation accuracy if it is exploiting SITE identity as an
    easier, more separable shortcut than the true PD/HC biomarker —
    exactly the pattern observed (train accuracy climbing normally, val
    accuracy plateauing well above chance but far below single-site
    literature numbers, persisting across normalization/regularization/
    capacity changes that would otherwise have fixed ordinary memorization).
    """
    def __init__(self, in_dim: int, n_sites: int, hidden: int = 32):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_sites),
        )

    def set_lambda(self, lambda_: float):
        self.grl.lambda_ = lambda_

    def forward(self, x):
        return self.net(self.grl(x))


# ── utility blocks ─────────────────────────────────────────────────────────────

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm1d(nn.Module):
    """LayerNorm over the last dimension (channel axis after permute)."""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):          # x: (B, T, C)
        return self.ln(x)


def gap1d(x):
    """Global average pool over the time dimension. (B,T,C) → (B,C)"""
    return x.mean(dim=1)


# ──────────────────────────────────────────────────────────────────────────────
#  INPUT NORMALISATION 
# ──────────────────────────────────────────────────────────────────────────────

class InputNorm(nn.Module):
    """
    LayerNorm on raw input (B,C,T), then permute to (B,T,C).
    LayerNorm(32) = 2*32 = 64 params.
    """
    def __init__(self, n_channels=32):
        super().__init__()
        self.ln = nn.LayerNorm(n_channels)   # normalises over channel dim

    def forward(self, x):                   # x: (B, 32, 1024)
        x = x.permute(0, 2, 1)             # (B, 1024, 32)
        x = self.ln(x)
        return x                            # (B, T, 32)


# ──────────────────────────────────────────────────────────────────────────────
# FRACTAL MULTI-SCALE CONVOLUTION  (3 hierarchical blocks)
# ──────────────────────────────────────────────────────────────────────────────

class FractalBlock(nn.Module):
    """
    Three parallel dilated Conv1Ds at different scales, then 1×1 fusion.

    scales = [s1, s2, s3]:
      Fs  = Conv1D(in,  n_inner, kernel_size=3, dilation=s)
      out = Swish(Conv1D([F_s1; F_s2; F_s3], out_filters, 1))
    """
    def __init__(self, in_ch, out_ch, scales, n_inner=None):
        super().__init__()
        if n_inner is None:
            n_inner = out_ch // len(scales)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, n_inner, kernel_size=3,
                      dilation=s, padding=s)        # same-length output
            for s in scales
        ])
        self.fusion = nn.Conv1d(n_inner * len(scales), out_ch, 1)
        self.norm   = nn.LayerNorm(out_ch)
        self.act    = Swish()

    def forward(self, x):          # x: (B, T, in_ch)
        x_t = x.permute(0, 2, 1)  # (B, in_ch, T)
        feats = [conv(x_t) for conv in self.convs]
        cat   = torch.cat(feats, dim=1)   # (B, n_inner*3, T)
        out   = self.fusion(cat)          # (B, out_ch, T)
        out   = out.permute(0, 2, 1)      # (B, T, out_ch)
        out   = self.norm(out)
        out   = self.act(out)
        return out


class MultiscaleFoundation(nn.Module):
    """
    Three cascaded FractalBlocks:
      L1: 32 → 64  scales [1,2,3]
      L2: 64 → 96  scales [2,4,8]
      L3: 96 → 96  scales [4,8,16]

    Full-width configuration (matches the original/pre-redesign 40-channel
    NHRN's widths exactly, e.g. 1.03M total parameters), adapted only at
    the input: n_channels=32 (was 40) since this is the 32-common-channel
    variant with no regional-aggregation block (see preprocessing_32.py).
    Every module keeps its original name and role in the computation graph.
    """
    def __init__(self, n_channels: int = 32):
        super().__init__()
        self.l1 = FractalBlock(n_channels, 64, scales=[1, 2, 3],  n_inner=20)
        self.l2 = FractalBlock(64,  96,  scales=[2, 4, 8],  n_inner=28)
        self.l3 = FractalBlock(96,  96,  scales=[4, 8, 16], n_inner=32)

    def forward(self, x):          # x: (B, T, 32)
        y1 = self.l1(x)            # (B, T, 64)
        y2 = self.l2(y1)           # (B, T, 96)
        y3 = self.l3(y2)           # (B, T, 96)
        return y1, y2, y3          # return all for potential skip connections


# ──────────────────────────────────────────────────────────────────────────────
# SYNAPTIC MODULATION  (strength + plasticity + modulation gates)
# ──────────────────────────────────────────────────────────────────────────────

class SynapticGate(nn.Module):
    """
    Equations (6–9):
      S    = σ(Ws · Yfractal)
      P    = tanh(Wp · Yfractal)
      M    = Wm · Yfractal
      Ysyn = M⊙S + Yfractal⊙P⊙S
    Each of Ws, Wp, Wm is Conv1d(dim, dim, 1) ≈ Linear.
    """
    def __init__(self, dim=96):
        super().__init__()
        self.strength   = nn.Linear(dim, dim)   # Ws
        self.plasticity = nn.Linear(dim, dim)   # Wp
        self.modulation = nn.Linear(dim, dim)   # Wm

    def forward(self, x):           # x: (B, T, dim)
        S    = torch.sigmoid(self.strength(x))
        P    = torch.tanh(self.plasticity(x))
        M    = self.modulation(x)
        return M * S + x * P * S   # Ysyn: (B, T, dim)


# ──────────────────────────────────────────────────────────────────────────────
# NEURAL OSCILLATION EXTRACTOR  (5 frequency bands)
# ──────────────────────────────────────────────────────────────────────────────

class NeuralOscillations(nn.Module):
    """
    5 band-specific Conv1Ds on Ysyn, stacked → MHA → GAP → feature vector.
    Bands: δ, θ, α, β, γ  (Bo = 5)
    """
    def __init__(self, in_dim=96, n_bands=5, band_dim=48, n_heads=4):
        super().__init__()
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, band_dim, kernel_size=k, padding=k // 2),
                nn.GroupNorm(_valid_groups(band_dim), band_dim),
                Swish()
            )
            for k in [3, 5, 9, 15, 31]   # ≈ δ,θ,α,β,γ temporal scales
        ])
        self.n_bands = n_bands
        self.band_dim = band_dim
        self.mha = nn.MultiheadAttention(embed_dim=band_dim,
                                         num_heads=n_heads,
                                         dropout=0.1,
                                         batch_first=True)
        self.norm = nn.LayerNorm(band_dim)

    def forward(self, x):           # x: (B, T, in_dim)
        xt = x.permute(0, 2, 1)    # (B, in_dim, T)
        bands = []
        for conv in self.band_convs:
            b = conv(xt)            # (B, band_dim, T)
            b = b.mean(dim=-1)      # GAP → (B, band_dim)
            bands.append(b)
        stacked = torch.stack(bands, dim=1)   # (B, 5, band_dim) — "sequence of bands"
        out, _  = self.mha(stacked, stacked, stacked)
        out     = self.norm(out)
        return out.mean(dim=1)      # (B, band_dim)  — oscillation feature vector


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE RECEPTIVE FIELD  (small / medium / large kernels)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveReceptiveField(nn.Module):
    """
    Three parallel Conv1Ds (k=3,7,15) with a learned soft-selector.
    .
    """
    def __init__(self, dim=96, out_dim=96):
        super().__init__()
        # RF Controller  (dim → 3 weights via Softmax)
        self.rf_ctrl = nn.Linear(dim, 3)

        # Three receptive field convolutions
        self.conv_s = nn.Conv1d(dim, out_dim, kernel_size=3,  padding=1)
        self.conv_m = nn.Conv1d(dim, out_dim, kernel_size=7,  padding=3)
        self.conv_l = nn.Conv1d(dim, out_dim, kernel_size=15, padding=7)

        self.norm = nn.LayerNorm(out_dim)
        self.act  = Swish()

    def forward(self, x):            # x: (B, T, dim)
        z = gap1d(x)                 # (B, dim)
        w = F.softmax(self.rf_ctrl(z), dim=-1)   # (B, 3)

        xt = x.permute(0, 2, 1)     # (B, dim, T)
        f_s = self.conv_s(xt).permute(0, 2, 1)   # (B, T, out_dim)
        f_m = self.conv_m(xt).permute(0, 2, 1)
        f_l = self.conv_l(xt).permute(0, 2, 1)

        # Weighted sum
        out = (w[:, 0:1, None] * f_s +
               w[:, 1:2, None] * f_m +
               w[:, 2:3, None] * f_l)
        return self.act(self.norm(out))   # (B, T, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL BIFURCATION  (fast / medium / slow pathways)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalBifurcation(nn.Module):
    """
    Three parallel temporal convolutions with increasing depth/kernel,
    then cross-path attention and merger.
    """
    def __init__(self, in_dim=96, hidden=64):
        super().__init__()
        # Fast path: small kernel
        self.fast = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=3,  padding=1),
            nn.GroupNorm(_valid_groups(hidden), hidden), Swish(),
            nn.Conv1d(hidden, hidden, kernel_size=3,  padding=1),
            nn.GroupNorm(_valid_groups(hidden), hidden), Swish(),
        )
        # Medium path
        self.medium = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=7,  padding=3),
            nn.GroupNorm(_valid_groups(hidden), hidden), Swish(),
            nn.Conv1d(hidden, hidden, kernel_size=7,  padding=3),
            nn.GroupNorm(_valid_groups(hidden), hidden), Swish(),
        )
        # Slow path: large kernel
        self.slow = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=15, padding=7),
            nn.GroupNorm(_valid_groups(hidden), hidden), Swish(),
            nn.Conv1d(hidden, hidden, kernel_size=15, padding=7),
            nn.GroupNorm(_valid_groups(hidden), hidden), Swish(),
        )

        # Cross-path attention (merges 3×hidden → hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=0.1, batch_first=True)
        self.path_merger = nn.Linear(3 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):           # x: (B, T, in_dim)
        xt = x.permute(0, 2, 1)    # (B, in_dim, T)

        f  = self.fast(xt).permute(0, 2, 1)    # (B, T, hidden)
        m  = self.medium(xt).permute(0, 2, 1)
        s  = self.slow(xt).permute(0, 2, 1)

        # Cross-path attention: query=fast, key=value=concat
        kv  = torch.cat([f, m, s], dim=1)       # (B, 3T, hidden)
        attn_out, _ = self.cross_attn(f, kv, kv)

        # Path merger
        merged = self.path_merger(
            torch.cat([f, m, s], dim=-1))        # (B, T, hidden)
        return self.norm(merged + attn_out)


# ──────────────────────────────────────────────────────────────────────────────
# COGNITIVE LOAD BALANCER
# ──────────────────────────────────────────────────────────────────────────────

class CognitiveLoadBalancer(nn.Module):
    """
    GAP → load estimator → softmax weights → weighted sum
    of multiple dense transformations applied to the global feature.
    """
    def __init__(self, in_dim, n_components=4):
        super().__init__()
        self.load_est  = nn.Linear(in_dim, n_components, bias=True)
        self.processors = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, in_dim), Swish())
            for _ in range(n_components)
        ])
        self.aggregator = nn.Linear(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):           # x: (B, T, in_dim)
        z      = gap1d(x)           # (B, in_dim)
        alpha  = F.softmax(self.load_est(z), dim=-1)   # (B, n_comp)
        comps  = torch.stack([proc(z) for proc in self.processors], dim=1)  # (B, n, D)
        balanced = (alpha.unsqueeze(-1) * comps).sum(dim=1)   # (B, D)
        out    = self.aggregator(balanced)
        return self.norm(out)       # (B, in_dim)


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY CONSOLIDATION  (GRU + forgetting gates)
# ──────────────────────────────────────────────────────────────────────────────

class MemoryConsolidation(nn.Module):
    def __init__(self, in_dim, hidden=128, use_gru: bool = False):
        super().__init__()
        self.use_gru = use_gru
        if use_gru:
            self.gru     = nn.GRU(in_dim, hidden, batch_first=True,
                                  bidirectional=False)
            # Forgetting / consolidation gate
            self.forget  = nn.Linear(hidden, hidden)
        else:
            self.fallback = nn.Sequential(nn.Linear(in_dim, hidden), Swish())
        self.norm    = nn.LayerNorm(hidden)

    def forward(self, x):            # x: (B, T, in_dim)
        if not self.use_gru:
            Ymem = self.fallback(x.mean(dim=1))   # GAP1D -> Linear, (B, hidden)
            return self.norm(Ymem)

        H, hT = self.gru(x)          # H: (B, T, hidden), hT: (1, B, hidden)
        hT = hT.squeeze(0)           # (B, hidden)

        # Gated consolidation: Supd = σ(forget(hT)) * H_mean
        h_mean = H.mean(dim=1)
        gate   = torch.sigmoid(self.forget(hT))
        Supd   = gate * h_mean + (1 - gate) * hT

        # Yshort-term + consolidated
        Ymem   = (H + Supd.unsqueeze(1)).mean(dim=1)  # GAP
        return self.norm(Ymem)       # (B, hidden)


# ──────────────────────────────────────────────────────────────────────────────
# SPECTRAL-TEMPORAL BRIDGE
# ──────────────────────────────────────────────────────────────────────────────

class SpectralTemporalBridge(nn.Module):
    """
    Conv1D time features × Dense freq projection → MHA → GAP+Dense.
    """
    def __init__(self, in_dim=96, out_dim=64, n_heads=4):
        super().__init__()
        self.time_proj = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        self.freq_proj = nn.Linear(in_dim, out_dim)
        self.mha       = nn.MultiheadAttention(out_dim, n_heads, dropout=0.1, batch_first=True)
        self.dense     = nn.Linear(out_dim, out_dim)
        self.norm      = nn.LayerNorm(out_dim)
        self.act       = Swish()

    def forward(self, x):                        # x: (B, T, in_dim)
        xt = x.permute(0, 2, 1)                 # (B, in_dim, T)
        t_feat = self.time_proj(xt).permute(0, 2, 1)  # (B, T, out_dim) — time domain
        f_feat = self.freq_proj(x)               # (B, T, out_dim)   — freq domain

        out, _ = self.mha(t_feat, f_feat, f_feat)
        out    = gap1d(out)                       # (B, out_dim)
        return self.act(self.norm(self.dense(out)))


# ──────────────────────────────────────────────────────────────────────────────
# EMERGENT PATTERN SYNTHESIS  (3 levels)
# ──────────────────────────────────────────────────────────────────────────────

class EmergentPatterns(nn.Module):
    """
    Yemergent = Swish(Dense(Epattern ⊙ σ(Egate)))
    Three cascaded emergence levels with gated transformations.
    """
    def __init__(self, in_dim, dims=(64, 32, 16)):
        super().__init__()
        layers = []
        d_in = in_dim
        for d_out in dims:
            layers.append(nn.ModuleDict({
                'pattern': nn.Linear(d_in, d_out),
                'gate':    nn.Linear(d_in, d_out),
                'norm':    nn.LayerNorm(d_out),
            }))
            d_in = d_out
        self.levels = nn.ModuleList([
            nn.ModuleDict({'pattern': l['pattern'],
                           'gate':    l['gate'],
                           'norm':    l['norm']})
            for l in layers
        ])
        self.dims = dims
        self.synthesis = nn.Linear(dims[-1], dims[-1])

    def forward(self, x):            # x: (B, in_dim)
        h = x
        for level in self.levels:
            E_pattern = level['pattern'](h)
            E_gate    = torch.sigmoid(level['gate'](h))
            h         = F.silu(level['norm'](E_pattern * E_gate))
        return self.synthesis(h)     # (B, dims[-1])


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE CRYSTALLIZER  (4 crystal stages)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureCrystallizer(nn.Module):
    """
    Ycrystal = Swish(Dense([S0_stable;...;S3_stable], 64))
    Progressive refinement through 4 stages.
    """
    def __init__(self, in_dim):
        super().__init__()
        # 4 stages: nucleation → growth → refinement → fusion
        self.nucleation  = nn.Linear(in_dim, 32)
        self.growth      = nn.Linear(32, 16)
        self.refinement  = nn.Linear(16, 16)
        # fusion input = s0(32) + s2(16) = 48
        self.fusion      = nn.Linear(48, 32)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in [32, 16, 16, 32]])

    def forward(self, x):            # x: (B, in_dim)
        s0 = F.silu(self.norms[0](self.nucleation(x)))   # (B, 32)
        s1 = F.silu(self.norms[1](self.growth(s0)))      # (B, 16)
        s2 = F.silu(self.norms[2](self.refinement(s1)))  # (B, 16)
        s3 = F.silu(self.norms[3](
            self.fusion(torch.cat([s0, s2], dim=-1))))    # (B, 32) via skip
        return s3                    # (B, 32)


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-MODAL RESONANCE & CHANNEL WEIGHTING
# ──────────────────────────────────────────────────────────────────────────────

class ResonanceFusion(nn.Module):
    """
    Yres = V ⊙ Softmax(q·k)
    Query/Key/Value projections on the concatenated high-level representation,
    followed by channel attention and multi-scale integration.
    """
    def __init__(self, in_dim, attn_dim=16):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.k = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(in_dim, attn_dim)

        # Channel attention
        self.ch_attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.Sigmoid()
        )

        # Multi-scale integration
        self.scale1 = nn.Linear(attn_dim, 8)
        self.scale2 = nn.Linear(8, 2)
        self.scale3 = nn.Linear(2, 1)

        self.norm = nn.LayerNorm(attn_dim)

    def forward(self, x):            # x: (B, in_dim)  — already pooled
        # Treat the single feature as a 1-token sequence for resonance
        q = self.q(x)               # (B, attn_dim)
        k = self.k(x)
        v = self.v(x)

        # Scaled dot product (single token → scalar gate)
        scale = math.sqrt(q.shape[-1])
        attn  = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / scale)

        res   = v * attn             # (B, attn_dim)
        res   = self.norm(res)

        # Channel attention
        ch_w  = self.ch_attn(res)
        res   = res * ch_w

        # Multi-scale integration (progressive bottleneck)
        s1    = F.silu(self.scale1(res))
        s2    = F.silu(self.scale2(s1))
        s3    = self.scale3(s2)

        return res, s3               # (B, attn_dim), (B, 1) confidence


# ──────────────────────────────────────────────────────────────────────────────
# FULL NHRN MODEL
# ──────────────────────────────────────────────────────────────────────────────

class NHRN(nn.Module):
    """
    Neuromorphic Hierarchical Resonance Network.

    Input : (B, 32, 1024)   — batch × channels × time samples (32-channel common-montage variant)
    Output: (B,)            — PD probability [0, 1]
    """

    def __init__(self,
                 n_channels : int = 32,
                 n_times    : int = 1024,
                 dropout    : float = 0.3,
                 feat_dropout: float = 0.1,
                 use_memory_gru: bool = False,
                 use_site_adversarial: bool = True,
                 n_sites: int = 3):
        super().__init__()
        self.use_site_adversarial = use_site_adversarial

        # Input normalisation ──────────────────────────────────────────
        self.input_norm = InputNorm(n_channels)

        # Multi-scale fractal foundation ──────────────────────────────
        self.fractal = MultiscaleFoundation(n_channels=n_channels)
        # Yfractal = Y3 ∈ (B, T, 96)

        # Synaptic gate + Neural oscillations ─────────────────────
        self.synaptic   = SynapticGate(dim=96)
        self.oscillation = NeuralOscillations(in_dim=96, n_bands=5,
                                              band_dim=48, n_heads=4)
        # Yosc ∈ (B, 48)

        # Adaptive receptive field ────────────────────────────────────
        self.arf = AdaptiveReceptiveField(dim=96, out_dim=96)
        # Yadapt ∈ (B, T, 96)

        # Temporal bifurcation ────────────────────────────────────────
        self.bifurcation = TemporalBifurcation(in_dim=96, hidden=64)
        # Ybif ∈ (B, T, 64)

        # Cognitive load balancer ─────────────────────────────────────
        self.cog_load = CognitiveLoadBalancer(in_dim=64, n_components=4)
        # Ybalanced ∈ (B, 64)

        # Memory consolidation ────────────────────────────────────────
        self.memory = MemoryConsolidation(in_dim=96, hidden=128, use_gru=use_memory_gru)
        # Ymem ∈ (B, 128)

        #  Spectral-temporal bridge ────────────────────────────────────
        self.spectral_bridge = SpectralTemporalBridge(in_dim=96,
                                                      out_dim=64, n_heads=4)
        # Yspec ∈ (B, 64)

        # ── Multi-level feature fusion ──────────────────────────────────────
        # concatenate: Yosc(48) + Ybalanced(64) + Ymem(128) + Yspec(64) = 304
        # (full-width configuration — matches the original 40-channel NHRN's
        # parameter budget; only the input layer differs, 32 vs 40 channels)
        fused_dim = 48 + 64 + 128 + 64   # 304

        # Emergent pattern synthesis ─────────────────────────────────
        self.emergent = EmergentPatterns(in_dim=fused_dim, dims=(64, 32, 16))
        # Yemergent ∈ (B, 16)

        #  Feature crystallizer ───────────────────────────────────────
        self.crystal = FeatureCrystallizer(in_dim=fused_dim)
        # Ycrystal ∈ (B, 32)

        # Combined before resonance: emergent(16) + crystal(32) = 48
        res_in = 16 + 32   # 48

        # Cross-modal resonance ──────────────────────────────────────
        self.resonance = ResonanceFusion(in_dim=res_in, attn_dim=16)
        # Yres ∈ (B, 16)  — unchanged: this is the classifier's fixed input width

        # Site-adversarial head (domain-adversarial training): tries to
        # predict recording site from the SAME Yres feature the main
        # classifier uses, via a gradient-reversal layer, so the shared
        # representation is pushed toward site-invariance instead of
        # exploiting site identity as a shortcut for the PD/HC decision.
        # Auxiliary only — has no effect on inference (forward() ignores it
        # unless site logits are explicitly requested during training).
        if self.use_site_adversarial:
            self.site_discriminator = SiteDiscriminator(in_dim=16, n_sites=n_sites)

        #  Classification head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.dropout = nn.Dropout(dropout)

        # Backbone regularization: every substantial feature-extraction
        # module upstream of Yfused (fractal conv, synaptic gates,
        # oscillation MHA, adaptive receptive field, temporal bifurcation,
        # GRU memory, spectral bridge) previously had NO dropout anywhere —
        # only the already-fused feature vector and classifier head did.
        # That left ~90% of the model's parameters, including a GRU that
        # readily memorizes exact temporal patterns, completely
        # unregularized against overfitting to individual subjects. This
        # single shared, parameter-free Dropout module is applied at each
        # major branch output in forward() to close that gap.
        self.feat_dropout = nn.Dropout(feat_dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_site_logits: bool = False):            # x: (B, 32, 1024)
        # Input normalisation
        x0 = self.input_norm(x)      # (B, T, 40)

        # Fractal multi-scale foundation
        _, _, Yfractal = self.fractal(x0)   # (B, T, 96)
        # Single backbone dropout point (not compounded across every branch
        # below): regularizes the shared trunk that every downstream module
        # (synaptic, oscillation, ARF, bifurcation, memory, spectral bridge)
        # derives from, without stacking dropout 5x deep along the path to
        # the classifier.
        Yfractal = self.feat_dropout(Yfractal)

        # Synaptic modulation
        Ysyn = self.synaptic(Yfractal)      # (B, T, 96)

        # Neural oscillation (uses synaptically gated signal)
        Yosc = self.oscillation(Ysyn)       # (B, 48)

        # Adaptive receptive field (uses raw Yfractal)
        Yadapt = self.arf(Yfractal)         # (B, T, 96)

        # Temporal bifurcation
        Ybif = self.bifurcation(Yadapt)     # (B, T, 64)

        # Cognitive load balancer
        Ybalanced = self.cog_load(Ybif)     # (B, 64)

        # Memory consolidation (uses Yadapt for temporal context)
        Ymem = self.memory(Yadapt)          # (B, 128)

        # Spectral-temporal bridge (uses Yfractal)
        Yspec = self.spectral_bridge(Yfractal)  # (B, 64)

        # Multi-level fusion
        Yfused = torch.cat([Yosc, Ybalanced, Ymem, Yspec], dim=-1)  # (B, 304)
        Yfused = self.dropout(Yfused)

        # Emergent patterns
        Yemergent = self.emergent(Yfused)   # (B, 16)

        # Feature crystallization
        Ycrystal = self.crystal(Yfused)     # (B, 32)

        # Combine and apply resonance fusion
        Ycombined = torch.cat([Yemergent, Ycrystal], dim=-1)  # (B, 48)
        Yres, _   = self.resonance(Ycombined)   # (B, 16)

        # Classification uses the resonance-refined representation.
        out = self.classifier(Yres)         # (B, 1)
        pPD = torch.sigmoid(out).squeeze(-1)  # (B,)

        if return_site_logits:
            if not self.use_site_adversarial:
                raise RuntimeError(
                    "return_site_logits=True but the model was built with "
                    "use_site_adversarial=False."
                )
            site_logits = self.site_discriminator(Yres)  # (B, n_sites)
            return pPD, site_logits
        return pPD

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def parameter_breakdown(self):
        """Print per-module parameter counts."""
        print(f"\n{'Component':<35} {'Params':>10}")
        print("─" * 47)
        components = {
            'InputNorm':          self.input_norm,
            'FractalL1':          self.fractal.l1,
            'FractalL2':          self.fractal.l2,
            'FractalL3':          self.fractal.l3,
            'SynapticStrength':   self.synaptic.strength,
            'SynapticPlasticity': self.synaptic.plasticity,
            'SynapticModulation': self.synaptic.modulation,
            'NeuralOscConv':      self.oscillation.band_convs,
            'OscillationMHA':     self.oscillation.mha,
            'AdaptiveRF':         self.arf,
            'TemporalBifurcation':self.bifurcation,
            'CognitiveLoad':      self.cog_load,
            'MemoryConsolidation':self.memory,
            'SpectralBridge':     self.spectral_bridge,
            'EmergentPatterns':   self.emergent,
            'FeatureCrystallizer':self.crystal,
            'ResonanceFusion':    self.resonance,
            'Classifier':         self.classifier,
        }
        if self.use_site_adversarial:
            components['SiteDiscriminator (aux)'] = self.site_discriminator
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:<33} {n:>10,}")
            total += n
        print("─" * 47)
        print(f"  {'TOTAL':<33} {total:>10,}\n")
        return total


# ── loss & optimizer helpers ───────────────────────────────────────────────────

def build_model(device='cpu', dropout=0.3, feat_dropout=0.1, use_memory_gru=False,
                 use_site_adversarial=True, n_sites=3):
    model = NHRN(n_channels=32, n_times=1024,
                 dropout=dropout, feat_dropout=feat_dropout,
                 use_memory_gru=use_memory_gru,
                 use_site_adversarial=use_site_adversarial,
                 n_sites=n_sites).to(device)
    return model


def build_optimizer(model, lr=1e-3, weight_decay=1e-3):
    """AdamW with weight decay """
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)


def binary_cross_entropy_loss(pred, target):
    """Equation (24): binary cross-entropy."""
    return F.binary_cross_entropy(pred, target.float())


# ── quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = build_model(device=device)
    model.parameter_breakdown()

    total, trainable = model.count_parameters()
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Forward pass test
    B = 4
    x = torch.randn(B, 32, 1024).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}  (values: {out.detach().cpu().numpy().round(4)})")
    print("Forward pass OK ✓")
