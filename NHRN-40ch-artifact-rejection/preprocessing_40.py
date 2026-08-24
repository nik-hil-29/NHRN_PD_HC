"""
preprocessing.py
=================
Unified UCSD (ds002778) + Iowa (IowaDataset) + UNM (UNMDataset) EEG
preprocessing for NHRN PD classification.

Channel harmonization (verified empirically/from BIDS metadata, not assumed):
  - CANONICAL_SCALP_CHANNELS: 32 electrodes present in all three sites
    (verified against UCSD's channels.tsv, Iowa's .vhdr headers, and UNM's
    Channel_location field). This block occupies model input channels 0-31
    for every site, in a fixed order.
  - Channels 32-39 ("the aggregate block") differ by site:
      * UCSD: no additional scalp electrodes exist. Instead, its 8 BioSemi
        external channels (EXG1-EXG8) are used directly here.
        ICA ocular-component detection for UCSD still uses the
        Fp1/Fp2 frontal proxy (see ICA_EOG_PROXY_CHANNELS), unchanged.
      * Iowa/UNM (63-66 channel caps): the ~31 scalp electrodes beyond the
        32-channel common montage ARE labelled with real 10-10 coordinates.
        Rather than discard them, they are spatially grouped into 8 regions
        (REGION_MEMBERS) and averaged into 8 additional channels:
            z_k(t) = (1/|G_k|) * sum_{c in G_k} x_c(t),   k = 1..8
  This block therefore carries real, site-specific signal for every site
  (EXG for UCSD, regional cortical averages for Iowa/UNM) rather than being
  zero anywhere; see AUX_BLOCK_START in train_NHRN, which applies
  "aux_block_dropout" so the model can't use this block's signal
  statistics as an implicit site cue.

Model input is therefore fixed at (40, 1024) regardless of site; no
architecture change is required.

File formats handled directly (no conversion step required):
  UCSD  : BIDS .bdf   (sub-*/ses-*/eeg/*.bdf)
  Iowa  : BrainVision  .vhdr/.eeg/.vmrk  (IowaDataset/Raw data/*.vhdr)
  UNM   : MATLAB v7.3 (HDF5) aggregated .mat files, one for ON-medication
          session, one for OFF-medication session (see UNM_ON_MAT/UNM_OFF_MAT)

UPDRS-III:
  UCSD : hard-coded ON/OFF table (paper Table 2)
  Iowa : merged from DataIowa.xlsx ('Rest' column = subject ID)
  UNM  : merged from IMPORT_ME_REST.xlsx (' UPDRS ON'/' UPDRS OFF' columns,
         keyed by PD_ID); no UPDRS for HC (matched-control demographics only)

Outputs (saved to ./preprocessed/):
  segments.npy    — shape (N, 40, 1024)  float32
  labels.npy      — shape (N,)           int   0=HC, 1=PD
  subjects.npy    — shape (N,)           str
  sessions.npy    — shape (N,)           str   'on' / 'off' / 'hc' / 'not_specified'
  sites.npy       — shape (N,)           str   'ucsd' / 'iowa' / 'unm'
  band_powers.npy — shape (N, 5)         Welch relative power per epoch
  session_band_powers.csv
  ica_component_log.csv
  channel_harmonization_log.csv  — per-recording: n canonical matched,
                                    which regional groups populated, any
                                    unmatched extra channels (for audit)
  metadata.pkl    — dict keyed by subject: label, updrs, per-session info

Artifact handling (identical across all three sites, with one exception):
  1. ICA fit on the continuous, 1-45 Hz filtered recording, over all real
     scalp channels available for that site (32 for UCSD, ~63 for Iowa,
     ~63 scalp + 1 EOG for UNM). UNM provides a dedicated vertical EOG
     channel (VEOG, channel 64) and this is used directly for ocular
     component detection via ICA.find_bads_eog. UCSD and Iowa have no
     documented EOG channel, so frontal electrodes (Fp1/Fp2) are used as
     an EOG-proxy instead. Muscle components via ICA.find_bads_muscle or a
     kurtosis fallback. Excluded component IDs logged to
     ica_component_log.csv, including which EOG source (real channel vs.
     frontal proxy) was used for each recording.
  2. Average reference computed over real scalp (eeg-typed) channels only
     — UNM's VEOG is excluded from the reference and from the
     canonical/regional-aggregate matching step, since it is not a
     cortical site.
  3. Channels are split into the 32-channel canonical block + 8-channel
     aggregate block (EXG1-8 for UCSD, regional averages for Iowa/UNM) to
     form the final 40-channel input.
  4. Amplitude-based epoch screening as a secondary safeguard
     (inspect_and_reject_artifact_epochs).
"""

import csv
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import h5py
from mne.preprocessing import ICA
from scipy.signal import welch
from scipy.stats import kurtosis

mne.set_log_level("WARNING")

# ── constants ────────────────────────────────────────────────────────────
FS          = 512
WIN_SEC     = 2
WIN_SAMPLES = FS * WIN_SEC
OVERLAP     = 0.5  
N_CHANNELS  = 40
N_SCALP_CHANNELS = 32
N_AGGREGATE_CHANNELS = 8

ARTIFACT_PEAK_TO_PEAK_UV = 1000.0
ARTIFACT_FLAT_UV = 0.5
ARTIFACT_RMS_UV = 250.0
ARTIFACT_MAX_BAD_CHANNEL_FRACTION = 0.30

ICA_APPLY = True
ICA_N_COMPONENTS = 20
ICA_METHOD = 'fastica'
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 'auto'
ICA_EOG_PROXY_CHANNELS = ['Fp1', 'Fp2'] 
UNM_EOG_CHANNEL = 'VEOG'  # UNM's 64th channel is a dedicated vertical EOG electrode
ICA_EOG_ZSCORE_THRESHOLD = 3.0
ICA_MUSCLE_KURTOSIS_THRESHOLD = 5.0
ICA_MAX_EXCLUDE_FRACTION = 0.5

BANDS = {
    'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0), 'low_gamma': (30.0, 45.0),
}
BAND_NAMES = tuple(BANDS.keys())

# 32 channels common to UCSD, Iowa, and UNM (verified: UCSD channels.tsv,
# Iowa .vhdr headers, UNM Channel_location).
CANONICAL_SCALP_CHANNELS = [
    'Fp1', 'Fp2', 'AF3', 'AF4',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO3', 'PO4',
    'O1', 'Oz', 'O2',
]
assert len(CANONICAL_SCALP_CHANNELS) == N_SCALP_CHANNELS

# UCSD's 8 BioSemi external channels, used directly as the aggregate block
# for that site (in place of the regional averages used at Iowa/UNM).
UCSD_EXG_CHANNELS = [f'EXG{i}' for i in range(1, 9)]
assert len(UCSD_EXG_CHANNELS) == N_AGGREGATE_CHANNELS

# 8 anatomical groups covering the ~31 extra 10-10 electrodes present on
# Iowa/UNM's 63-66 channel caps but absent from the 32-channel common
# montage. Only electrodes with a DOCUMENTED coordinate are grouped; any
# channel not matching one of these sets (and not in CANONICAL_SCALP_CHANNELS)
# is left out of the aggregate and logged, not silently guessed.
REGION_GROUP_NAMES = [
    'anterior_frontal', 'frontal_lateral', 'fronto_central', 'fronto_temporal',
    'central_lateral', 'temporo_parietal', 'parietal_lateral', 'parieto_occipital',
]
REGION_MEMBERS = {
    'anterior_frontal':  {'AF7', 'AF8', 'AFz', 'F1', 'F2'},
    'frontal_lateral':   {'F5', 'F6'},
    'fronto_central':    {'FC3', 'FC4', 'FCz'},
    'fronto_temporal':   {'FT7', 'FT8', 'FT9', 'FT10'},
    'central_lateral':   {'C1', 'C2', 'C5', 'C6'},
    'temporo_parietal':  {'CP3', 'CP4', 'CPz', 'TP7', 'TP8', 'TP9', 'TP10'},
    'parietal_lateral':  {'P1', 'P2', 'P5', 'P6'},
    'parieto_occipital': {'PO7', 'PO8', 'POz'},
}
assert len(REGION_GROUP_NAMES) == N_AGGREGATE_CHANNELS

UCSD_UPDRS = {
    'sub-pd9':  {'on': 16, 'off': 32}, 'sub-pd11': {'on': 20, 'off': 20},
    'sub-pd5':  {'on': 22, 'off': 28}, 'sub-pd26': {'on': 25, 'off': 42},
    'sub-pd16': {'on': 27, 'off': 32}, 'sub-pd17': {'on': 30, 'off': 30},
    'sub-pd19': {'on': 30, 'off': 38}, 'sub-pd14': {'on': 31, 'off': 38},
    'sub-pd3':  {'on': 33, 'off': 43}, 'sub-pd6':  {'on': 34, 'off': 40},
    'sub-pd23': {'on': 36, 'off': 44}, 'sub-pd22': {'on': 42, 'off': 58},
    'sub-pd12': {'on': 44, 'off': 47}, 'sub-pd28': {'on': 46, 'off': 47},
    'sub-pd13': {'on': 54, 'off': 49},
}

UNM_PD_IDS = ['801', '802', '804', '805', '806', '807', '808', '809', '810', '811',
              '813', '814', '815', '816', '817', '818', '819', '820', '821', '822',
              '823', '824', '825', '826', '827', '828', '829']
UNM_HC_IDS = ['890', '891', '892', '893', '894', '895', '896', '897', '898', '899',
              '900', '901', '902', '903', '904', '905', '906', '907', '908', '909',
              '910', '911', '912', '913', '914', '8070', '8060']


def canonical_channel_key(name: str) -> str:
    """Normalize EEG channel labels for matching across naming styles."""
    clean = name.strip()
    if clean.upper().startswith('EEG '):
        clean = clean[4:]
    clean = clean.split('-')[0].split('_')[0].split('.')[0]
    return ''.join(ch for ch in clean.lower() if ch.isalnum())


# ── artifact handling ───────────────────────────────────────────────────
def inspect_and_reject_artifact_epochs(
    segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reject epochs contaminated by movement/ocular, muscle, or flat channels."""
    if len(segments) == 0:
        report = {
            'epochs_before_artifact_rejection': 0, 'epochs_after_artifact_rejection': 0,
            'epochs_rejected_total': 0, 'epochs_rejected_movement_ocular': 0,
            'epochs_rejected_muscle': 0, 'epochs_rejected_flat': 0,
            'rejected_epoch_fraction': 0.0,
        }
        return segments, np.zeros((0,), dtype=bool), report

    peak_to_peak_uv = np.ptp(segments, axis=-1) * 1e6
    rms_uv = np.sqrt(np.mean(np.square(segments), axis=-1)) * 1e6

    movement_ocular_channels = peak_to_peak_uv > ARTIFACT_PEAK_TO_PEAK_UV
    muscle_channels = rms_uv > ARTIFACT_RMS_UV
    flat_channels = peak_to_peak_uv < ARTIFACT_FLAT_UV

    movement_ocular = movement_ocular_channels.mean(axis=1) > ARTIFACT_MAX_BAD_CHANNEL_FRACTION
    muscle = muscle_channels.mean(axis=1) > ARTIFACT_MAX_BAD_CHANNEL_FRACTION
    flat = flat_channels.mean(axis=1) > ARTIFACT_MAX_BAD_CHANNEL_FRACTION
    keep = ~(movement_ocular | muscle | flat)

    warning = None
    if not keep.any() and len(segments) > 0:
        warning = ("Artifact inspection marked every epoch. All epochs were retained "
                   "to avoid an empty subject/session; inspect this recording manually.")
        keep = np.ones(len(segments), dtype=bool)

    report = {
        'epochs_before_artifact_rejection': int(len(segments)),
        'epochs_after_artifact_rejection': int(keep.sum()),
        'epochs_rejected_total': int((~keep).sum()),
        'epochs_rejected_movement_ocular': int(movement_ocular.sum()),
        'epochs_rejected_muscle': int(muscle.sum()),
        'epochs_rejected_flat': int(flat.sum()),
        'rejected_epoch_fraction': float((~keep).mean()),
        'movement_ocular_threshold_uv': ARTIFACT_PEAK_TO_PEAK_UV,
        'muscle_rms_threshold_uv': ARTIFACT_RMS_UV,
        'flat_threshold_uv': ARTIFACT_FLAT_UV,
        'max_bad_channel_fraction': ARTIFACT_MAX_BAD_CHANNEL_FRACTION,
        'peak_to_peak_uv_p95': float(np.percentile(peak_to_peak_uv, 95)),
        'rms_uv_p95': float(np.percentile(rms_uv, 95)),
    }
    if warning is not None:
        report['warning'] = warning
    return segments[keep], keep, report


def run_ica_artifact_correction(
    raw: mne.io.Raw, subject: str, session: str,
    eog_channels: list[str] | None = None,
) -> tuple[mne.io.Raw, dict[str, object]]:
    """
    Fit ICA on the continuous filtered recording; remove ocular/muscle components.

    ``eog_channels``: real, dedicated EOG channel(s) for this recording, if
    any exist (e.g. UNM's VEOG). When given and present in ``raw``, these are
    used directly for ocular-component detection instead of the Fp1/Fp2
    frontal proxy — a real EOG channel gives a cleaner correlation target
    than a scalp proxy. Falls back to the proxy if none are present.
    """
    report: dict[str, object] = {
        'subject': subject, 'session': session, 'applied': False,
        'method': ICA_METHOD, 'random_state': ICA_RANDOM_STATE, 'n_components': 0,
        'eog_channel_source': None, 'eog_proxy_channels_used': [], 'eog_component_ids': [],
        'muscle_component_ids': [], 'excluded_component_ids': [],
        'n_excluded': 0, 'error': None,
    }
    if not ICA_APPLY:
        return raw, report

    n_components = min(ICA_N_COMPONENTS, len(raw.ch_names) - 1)
    report['n_components'] = int(n_components)

    try:
        ica = ICA(n_components=n_components, method=ICA_METHOD,
                   random_state=ICA_RANDOM_STATE, max_iter=ICA_MAX_ITER)
        ica.fit(raw, verbose=False)

        eog_inds: list[int] = []
        real_eog_present = [ch for ch in (eog_channels or []) if ch in raw.ch_names]
        proxy_present = [ch for ch in ICA_EOG_PROXY_CHANNELS if ch in raw.ch_names]
        eog_source = real_eog_present if real_eog_present else proxy_present
        report['eog_channel_source'] = 'real_eog' if real_eog_present else ('frontal_proxy' if proxy_present else None)
        if eog_source:
            try:
                found, _scores = ica.find_bads_eog(
                    raw, ch_name=eog_source, threshold=ICA_EOG_ZSCORE_THRESHOLD, verbose=False)
                eog_inds = list(dict.fromkeys(int(i) for i in found))
            except Exception as eog_err:
                report['eog_detection_error'] = str(eog_err)
        report['eog_proxy_channels_used'] = eog_source

        muscle_inds: list[int] = []
        try:
            found = ica.find_bads_muscle(raw, verbose=False)
            if isinstance(found, tuple):
                found = found[0]
            muscle_inds = list(dict.fromkeys(int(i) for i in found))
        except AttributeError:
            sources = ica.get_sources(raw).get_data()
            comp_kurtosis = kurtosis(sources, axis=1, fisher=True)
            muscle_inds = [int(i) for i, k in enumerate(comp_kurtosis)
                            if k > ICA_MUSCLE_KURTOSIS_THRESHOLD and i not in eog_inds]
        except Exception as muscle_err:
            report['muscle_detection_error'] = str(muscle_err)

        exclude = sorted(set(eog_inds) | set(muscle_inds))
        max_exclude = max(1, int(np.floor(n_components * ICA_MAX_EXCLUDE_FRACTION)))
        if len(exclude) > max_exclude:
            report['exclusion_capped'] = True
            exclude = exclude[:max_exclude]

        ica.exclude = exclude
        cleaned = raw.copy()
        ica.apply(cleaned, verbose=False)

        report.update({'applied': True, 'eog_component_ids': eog_inds,
                        'muscle_component_ids': muscle_inds,
                        'excluded_component_ids': exclude, 'n_excluded': len(exclude)})
        return cleaned, report
    except Exception as e:
        report['error'] = str(e)
        report['warning'] = ("ICA fitting/application failed; recording passed through "
                              "un-corrected. Amplitude-based epoch rejection still applies.")
        return raw, report


def welch_relative_band_power(segments: np.ndarray, fs: int = FS) -> np.ndarray:
    """Channel-averaged Welch relative power for the five EEG bands."""
    if len(segments) == 0:
        return np.empty((0, len(BAND_NAMES)), dtype=np.float32)
    nperseg = min(fs, segments.shape[-1])
    noverlap = nperseg // 2
    freqs, psd = welch(segments, fs=fs, window='hamming', nperseg=nperseg,
                        noverlap=noverlap, axis=-1)
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.maximum(np.trapz(psd[..., total_mask], freqs[total_mask], axis=-1), 1e-12)
    band_features = []
    for low, high in BANDS.values():
        band_mask = (freqs >= low) & (freqs <= high) if high == 45.0 else (freqs >= low) & (freqs < high)
        band_power = np.trapz(psd[..., band_mask], freqs[band_mask], axis=-1)
        band_features.append((band_power / total_power).mean(axis=1))
    return np.stack(band_features, axis=1).astype(np.float32)


def segment(data: np.ndarray, win_samples: int = WIN_SAMPLES,
            stride: int | None = None) -> np.ndarray:
    """Slice a (channels, T) array into overlapping windows -> (n_win, channels, win_samples)."""
    if stride is None:
        stride = int(win_samples * (1 - OVERLAP))
    n_channels, n_times = data.shape
    if n_times < win_samples:
        return np.empty((0, n_channels, win_samples), dtype=data.dtype)
    starts = range(0, n_times - win_samples + 1, stride)
    return np.stack([data[:, s:s + win_samples] for s in starts], axis=0)


def z_score_normalize(segs: np.ndarray) -> np.ndarray:
    """Per-channel, per-epoch z-score normalization."""
    mean = segs.mean(axis=-1, keepdims=True)
    std = segs.std(axis=-1, keepdims=True) + 1e-8
    return ((segs - mean) / std).astype(np.float32)


# ── unified 40-channel harmonization ───────────────────────────────────────
def classify_extra_channel(name: str) -> str | None:
    """Map a non-canonical channel name to one of the 8 regional groups, or None."""
    for region, members in REGION_MEMBERS.items():
        if name in members:
            return region
    return None


def finalize_40ch(
    raw: mne.io.Raw, subject: str, session: str, site: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Shared pipeline for ALL three sites, given a Raw object already containing
    the real scalp EEG channels (plus, for UCSD, its 8 EXG channels typed
    'misc') at whatever native sampling rate/montage size the site provides.

    Steps: resample -> 1-45Hz filter -> ICA artifact correction -> average
    reference (over scalp EEG channels only) -> split into the fixed
    32-channel canonical block + 8-channel aggregate block (EXG for UCSD,
    regional averages for Iowa/UNM) -> concatenate to (40, T).
    """
    if round(raw.info['sfreq']) != FS:
        raw.resample(FS, npad='auto')
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', fir_window='hamming',
               skip_by_annotation='edge', verbose=False)

    eog_channels = [UNM_EOG_CHANNEL] if site == 'unm' else None
    raw, ica_report = run_ica_artifact_correction(raw, subject, session, eog_channels=eog_channels)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Exclude EOG-typed channels (e.g. UNM's VEOG) from the scalp-matching
    # step below — they're real, useful signal for ICA (already used above)
    # but are not a cortical site and shouldn't be aggregated or logged as
    # an "unmatched" scalp channel.
    ch_types = raw.get_channel_types()
    scalp_idx = [i for i, t in enumerate(ch_types) if t == 'eeg']
    ch_names = [raw.ch_names[i] for i in scalp_idx]
    data = raw.get_data(picks=scalp_idx).astype(np.float32)  # (n_ch, T)

    canonical_key_to_target = {canonical_channel_key(c): c for c in CANONICAL_SCALP_CHANNELS}
    present_by_key: dict[str, str] = {}
    for ch in ch_names:
        present_by_key.setdefault(canonical_channel_key(ch), ch)

    canon_block = np.zeros((N_SCALP_CHANNELS, data.shape[1]), dtype=np.float32)
    matched_source_names: set[str] = set()
    n_canonical_matched = 0
    for i, target in enumerate(CANONICAL_SCALP_CHANNELS):
        key = canonical_channel_key(target)
        src = present_by_key.get(key)
        if src is not None:
            canon_block[i] = data[ch_names.index(src)]
            matched_source_names.add(src)
            n_canonical_matched += 1

    agg_block = np.zeros((N_AGGREGATE_CHANNELS, data.shape[1]), dtype=np.float32)
    populated_groups: list[str] = []
    unmatched_extra: list[str] = []
    if site == 'ucsd':
        exg_idx = [i for i, t in enumerate(ch_types) if t == 'misc']
        exg_data = raw.get_data(picks=exg_idx).astype(np.float32) if exg_idx else None
        exg_names = [raw.ch_names[i] for i in exg_idx]
        for name, row in zip(exg_names, exg_data if exg_data is not None else []):
            agg_block[UCSD_EXG_CHANNELS.index(name)] = row
        populated_groups = exg_names
    else:
        for region_idx, region in enumerate(REGION_GROUP_NAMES):
            members = REGION_MEMBERS[region]
            idxs = [i for i, ch in enumerate(ch_names)
                    if ch in members and ch not in matched_source_names]
            if idxs:
                agg_block[region_idx] = data[idxs].mean(axis=0)
                populated_groups.append(region)
        for ch in ch_names:
            if ch not in matched_source_names and classify_extra_channel(ch) is None:
                unmatched_extra.append(ch)

    full = np.concatenate([canon_block, agg_block], axis=0)  # (40, T)
    diag = {
        'site': site, 'subject': subject, 'session': session,
        'n_canonical_matched': n_canonical_matched,
        'n_regional_groups_populated': len(populated_groups),
        'regional_groups_populated': ';'.join(populated_groups),
        'unmatched_extra_channels': ';'.join(unmatched_extra),
        'ica_report': ica_report,
    }
    return full, diag


# ── UCSD loader ─────────────────────────────────────────────────────────
def select_ucsd_scalp(raw: mne.io.Raw) -> mne.io.Raw | None:
    """Pick the 32 canonical scalp channels plus EXG1-8; drop Status and anything else."""
    available_by_key: dict[str, str] = {}
    for ch in raw.ch_names:
        available_by_key.setdefault(canonical_channel_key(ch), ch)

    pairs = []  # (source_name_in_file, canonical_target_name)
    for target in CANONICAL_SCALP_CHANNELS:
        src = available_by_key.get(canonical_channel_key(target))
        if src is not None:
            pairs.append((src, target))

    if len(pairs) < N_SCALP_CHANNELS:
        print(f"  [WARN] UCSD recording only matched {len(pairs)}/{N_SCALP_CHANNELS} "
              f"canonical channels")
        if not pairs:
            return None

    exg_present = [ch for ch in UCSD_EXG_CHANNELS if ch in raw.ch_names]
    if len(exg_present) < N_AGGREGATE_CHANNELS:
        print(f"  [WARN] UCSD recording only matched {len(exg_present)}/{N_AGGREGATE_CHANNELS} "
              f"EXG channels")

    raw.pick_channels([src for src, _ in pairs] + exg_present)
    rename_map = {src: target for src, target in pairs if src != target}
    if rename_map:
        raw.rename_channels(rename_map)
    if exg_present:
        # 'misc' keeps EXG out of the average-reference and ICA-fit picks
        # below (they're not cortical sites, so shouldn't be re-referenced
        # against scalp average or treated as scalp data for ICA).
        raw.set_channel_types({ch: 'misc' for ch in exg_present})
    return raw


def discover_ucsd_recordings(root: str | Path) -> list[dict[str, object]]:
    items = []
    for fpath in sorted(Path(root).rglob("*.bdf")):
        parts = fpath.stem.split('_')
        sub = next((p for p in parts if p.startswith('sub-')), None)
        ses = next((p.replace('ses-', '') for p in parts if p.startswith('ses-')), 'unknown')
        if sub is None:
            continue
        label = 'pd' if sub.startswith('sub-pd') else 'hc' if sub.startswith('sub-hc') else None
        if label is None:
            continue
        updrs = UCSD_UPDRS.get(sub, {}).get(ses) if label == 'pd' else None
        items.append({'site': 'ucsd', 'subject': sub, 'session': ses if label == 'pd' else 'hc',
                       'label': label, 'updrs_iii': updrs, 'path': str(fpath)})
    return items


def load_ucsd_raw(item: dict[str, object]) -> mne.io.Raw | None:
    raw = mne.io.read_raw_bdf(item['path'], preload=True, verbose=False)
    return select_ucsd_scalp(raw)


# ── Iowa loader ─────────────────────────────────────────────────────────
def discover_iowa_recordings(root: str | Path, updrs_xlsx: str | Path | None) -> list[dict[str, object]]:
    updrs_lookup: dict[str, dict[str, object]] = {}
    if updrs_xlsx is not None and Path(updrs_xlsx).exists():
        df = pd.read_excel(updrs_xlsx, sheet_name='ALL')
        df['Rest'] = df['Rest'].astype(str).str.strip()
        for _, row in df.iterrows():
            updrs_lookup[row['Rest']] = {
                'updrs_iii': row.get('UPDRS III'), 'moca': row.get('moCA'),
                'age': row.get('Age'), 'gender': row.get('Gender'),
                'disease_duration_yrs': row.get('Disease Duration'),
            }

    items = []
    for fpath in sorted(Path(root).rglob("*.vhdr")):
        stem = fpath.stem
        label = ('hc' if stem.lower().startswith('control')
                  else 'pd' if stem.lower().startswith('pd') else None)
        if label is None:
            continue
        extra = updrs_lookup.get(stem, {})
        items.append({'site': 'iowa', 'subject': f'iowa-{stem}', 'session': 'not_specified',
                       'label': label, 'updrs_iii': extra.get('updrs_iii'),
                       'moca': extra.get('moca'), 'age': extra.get('age'),
                       'gender': extra.get('gender'),
                       'disease_duration_yrs': extra.get('disease_duration_yrs'),
                       'path': str(fpath)})
    return items


def load_iowa_raw(item: dict[str, object]) -> mne.io.Raw | None:
    raw = mne.io.read_raw_brainvision(item['path'], preload=True, verbose=False)
    raw.pick_types(eeg=True, eog=False, misc=False, exclude=[])
    return raw


# ── UNM loader (HDF5/-v7.3 MAT, nested cell arrays) ────────────────────
def h5_str(dataset) -> str:
    return ''.join(chr(c) for c in np.array(dataset).flatten())


def open_unm_mat(mat_path: str) -> tuple[h5py.File, list[str], np.ndarray, list[int]]:
    """Returns (file_handle, channel_names, eeg_channel_refs, subjects_per_condition)."""
    f = h5py.File(mat_path, 'r')
    chan_refs = np.array(f['Channel_location']).flatten()
    channel_names = [h5_str(f[ref]) for ref in chan_refs]
    eeg_refs = np.array(f['EEG']).flatten()
    first_cond_refs = np.array(f[eeg_refs[0]]).flatten()
    subjects_per_condition = [f[first_cond_refs[c]].shape[0] for c in range(len(first_cond_refs))]
    return f, channel_names, eeg_refs, subjects_per_condition


def extract_unm_subject_data(f: h5py.File, eeg_refs: np.ndarray,
                              n_channels: int, condition_idx: int, subject_idx: int) -> np.ndarray:
    """Dereference every channel for one (condition, subject) -> (n_channels, n_samples) array, in uV."""
    vectors = []
    for ch in range(n_channels):
        cond_refs = np.array(f[eeg_refs[ch]]).flatten()
        subj_refs = np.array(f[cond_refs[condition_idx]]).flatten()
        vec = np.array(f[subj_refs[subject_idx]]).flatten().astype(np.float32)
        vectors.append(vec)
    min_len = min(len(v) for v in vectors)
    return np.stack([v[:min_len] for v in vectors], axis=0)


def discover_unm_recordings(on_mat: str, off_mat: str,
                             updrs_xlsx: str | Path | None) -> list[dict[str, object]]:
    clin_lookup: dict[str, dict[str, object]] = {}
    ctl_match: dict[str, dict[str, object]] = {}
    if updrs_xlsx is not None and Path(updrs_xlsx).exists():
        clin_df = pd.read_excel(updrs_xlsx, sheet_name='Sheet1')
        clin_df.columns = [c.strip() for c in clin_df.columns]
        clin_df['PD_ID'] = clin_df['PD_ID'].astype(str).str.strip()
        for _, row in clin_df.iterrows():
            clin_lookup[row['PD_ID']] = row.to_dict()
            ctl_id = str(row.get('MATCH CTL_ID', '')).strip()
            if ctl_id:
                ctl_match[ctl_id] = {'age': row.get('MATCH CTL_Age'), 'gender': row.get('MATCH CTL_Sex')}

    items = []
    for sub_id in UNM_PD_IDS:
        subject_idx = UNM_PD_IDS.index(sub_id)
        extra = clin_lookup.get(sub_id, {})
        for session, mat_path, updrs_key in [('on', on_mat, 'UPDRS ON'), ('off', off_mat, 'UPDRS OFF')]:
            items.append({
                'site': 'unm', 'subject': f'unm-{sub_id}', 'session': session, 'label': 'pd',
                'updrs_iii': extra.get(updrs_key), 'age': extra.get('PD_Age'),
                'gender': extra.get('PD_Sex'), 'disease_duration_yrs': extra.get('Yrs Dx'),
                'mat_path': mat_path, 'condition_idx': 0, 'subject_idx': subject_idx,
            })
    for sub_id in UNM_HC_IDS:
        subject_idx = UNM_HC_IDS.index(sub_id)
        extra = ctl_match.get(sub_id, {})
        items.append({
            'site': 'unm', 'subject': f'unm-{sub_id}', 'session': 'n/a', 'label': 'hc',
            'updrs_iii': None, 'age': extra.get('age'), 'gender': extra.get('gender'),
            'disease_duration_yrs': None,
            'mat_path': on_mat, 'condition_idx': 1, 'subject_idx': subject_idx,
        })
    return items


def load_unm_raw(item: dict[str, object], mat_cache: dict[str, tuple]) -> mne.io.Raw:
    """mat_cache: {mat_path: (file_handle, channel_names, eeg_refs)} to avoid re-opening per subject."""
    mat_path = item['mat_path']
    if mat_path not in mat_cache:
        f, channel_names, eeg_refs, _ = open_unm_mat(mat_path)
        mat_cache[mat_path] = (f, channel_names, eeg_refs)
    f, channel_names, eeg_refs = mat_cache[mat_path]

    data_uv = extract_unm_subject_data(f, eeg_refs, len(channel_names),
                                        item['condition_idx'], item['subject_idx'])
    ch_types = ['eog' if ch == UNM_EOG_CHANNEL else 'eeg' for ch in channel_names]
    info = mne.create_info(ch_names=channel_names, sfreq=500.0, ch_types=ch_types)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)  # uV -> V for MNE convention
    return raw


# ── CSV/log savers ──────────────────────────────────────────────────────
def save_session_band_power_csv(rows: list[dict[str, object]], out_dir: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path = os.path.join(out_dir, 'session_band_powers.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ica_component_log(reports: list[dict[str, object]], out_dir: str) -> None:
    if not reports:
        return
    fieldnames = ['site', 'subject', 'session', 'applied', 'method', 'random_state',
                  'n_components', 'eog_channel_source', 'eog_proxy_channels_used',
                  'eog_component_ids', 'muscle_component_ids', 'excluded_component_ids',
                  'n_excluded', 'error']
    path = os.path.join(out_dir, 'ica_component_log.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in reports:
            row = dict(r)
            for key in ('eog_proxy_channels_used', 'eog_component_ids',
                        'muscle_component_ids', 'excluded_component_ids'):
                row[key] = ';'.join(str(v) for v in row.get(key, []))
            writer.writerow(row)


def save_channel_harmonization_log(diags: list[dict[str, object]], out_dir: str) -> None:
    if not diags:
        return
    fieldnames = ['site', 'subject', 'session', 'n_canonical_matched',
                  'n_regional_groups_populated', 'regional_groups_populated',
                  'unmatched_extra_channels']
    path = os.path.join(out_dir, 'channel_harmonization_log.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(diags)


# ── master orchestration ────────────────────────────────────────────────
def preprocess_all(
    ucsd_root: str, iowa_root: str, iowa_updrs_xlsx: str,
    unm_on_mat: str, unm_off_mat: str, unm_updrs_xlsx: str,
    out_dir: str = './preprocessed',
    stride: int | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stride = stride if stride is not None else int(WIN_SAMPLES * (1 - OVERLAP))

    print("Discovering recordings...")
    ucsd_items = discover_ucsd_recordings(ucsd_root)
    iowa_items = discover_iowa_recordings(iowa_root, iowa_updrs_xlsx)
    unm_items = discover_unm_recordings(unm_on_mat, unm_off_mat, unm_updrs_xlsx)
    print(f"  UCSD: {len(ucsd_items)} recordings")
    print(f"  Iowa: {len(iowa_items)} recordings")
    print(f"  UNM : {len(unm_items)} recordings")

    all_segments, all_labels, all_subjects, all_sessions, all_sites = [], [], [], [], []
    all_band_rows, all_band_powers, ica_reports, harmonization_diags = [], [], [], []
    metadata: dict[str, dict[str, object]] = {}
    unm_mat_cache: dict[str, tuple] = {}

    def process_item(item: dict[str, object], raw: mne.io.Raw | None):
        if raw is None:
            print(f"  SKIP {item['subject']}/{item['session']} (could not load channels)")
            return
        try:
            data, diag = finalize_40ch(raw, item['subject'], item['session'], item['site'])
        except Exception as e:
            print(f"  ERROR {item['subject']}/{item['session']}: {e}")
            return
        harmonization_diags.append(diag)
        ica_reports.append(diag['ica_report'])

        segs = segment(data, WIN_SAMPLES, stride)
        segs, keep_mask, artifact_report = inspect_and_reject_artifact_epochs(segs)
        segs = z_score_normalize(segs)
        n = len(segs)
        if n == 0:
            print(f"  {item['subject']}/{item['session']}: 0 epochs after artifact rejection")
            return

        band_powers = welch_relative_band_power(segs)
        all_segments.append(segs)
        all_band_powers.append(band_powers)
        all_labels.extend([1 if item['label'] == 'pd' else 0] * n)
        all_subjects.extend([item['subject']] * n)
        all_sessions.extend([item['session']] * n)
        all_sites.extend([item['site']] * n)

        all_band_rows.append({
            'site': item['site'], 'subject': item['subject'], 'session': item['session'],
            'label': item['label'], 'n_segments': n,
            **{f'{b}_mean': float(band_powers[:, i].mean()) for i, b in enumerate(BAND_NAMES)},
        })

        meta = metadata.setdefault(item['subject'], {'site': item['site'], 'label': item['label'], 'sessions': []})
        meta['sessions'].append({
            'session': item['session'], 'n_segs': n, 'updrs_iii': item.get('updrs_iii'),
            'age': item.get('age'), 'gender': item.get('gender'),
            'disease_duration_yrs': item.get('disease_duration_yrs'),
            'artifact_inspection': artifact_report, 'channel_harmonization': diag,
        })
        print(f"  {item['subject']}/{item['session']}: {n} epochs "
              f"(canon={diag['n_canonical_matched']}/32, "
              f"regional_groups={diag['n_regional_groups_populated']}/8)")

    print("\n=== UCSD ===")
    for item in ucsd_items:
        raw = load_ucsd_raw(item)
        process_item(item, raw)

    print("\n=== Iowa ===")
    for item in iowa_items:
        raw = load_iowa_raw(item)
        process_item(item, raw)

    print("\n=== UNM ===")
    for item in unm_items:
        raw = load_unm_raw(item, unm_mat_cache)
        process_item(item, raw)
    for f, _, _ in unm_mat_cache.values():
        f.close()

    if not all_segments:
        raise RuntimeError("No segments were produced — check dataset paths.")

    segments = np.concatenate(all_segments, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=object)
    sessions = np.array(all_sessions, dtype=object)
    sites = np.array(all_sites, dtype=object)
    band_powers_all = np.concatenate(all_band_powers, axis=0) if all_band_powers else np.empty((0, len(BAND_NAMES)))

    np.save(os.path.join(out_dir, 'segments.npy'), segments)
    np.save(os.path.join(out_dir, 'labels.npy'), labels)
    np.save(os.path.join(out_dir, 'subjects.npy'), subjects)
    np.save(os.path.join(out_dir, 'sessions.npy'), sessions)
    np.save(os.path.join(out_dir, 'sites.npy'), sites)
    np.save(os.path.join(out_dir, 'band_powers.npy'), band_powers_all)
    np.save(os.path.join(out_dir, 'band_names.npy'), np.array(BAND_NAMES, dtype=object))
    with open(os.path.join(out_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    save_session_band_power_csv(all_band_rows, out_dir)
    save_ica_component_log(ica_reports, out_dir)
    save_channel_harmonization_log(harmonization_diags, out_dir)

    print(f"\n{'='*60}")
    print(f"Total segments: {len(segments)}  shape={segments.shape}")
    print(f"By site: {pd.Series(sites).value_counts().to_dict()}")
    print(f"By label: {pd.Series(labels).value_counts().to_dict()}")
    print(f"Saved to {out_dir}/")


def load_preprocessed(out_dir: str = './preprocessed', mmap: bool = False) -> dict:
    """Load previously saved preprocessed data."""
    segments_path = os.path.join(out_dir, 'segments.npy')
    mmap_mode = 'r' if mmap else None
    X = np.load(segments_path, allow_pickle=True, mmap_mode=mmap_mode)
    y = np.load(os.path.join(out_dir, 'labels.npy'), allow_pickle=True)
    S = np.load(os.path.join(out_dir, 'subjects.npy'), allow_pickle=True)
    E = np.load(os.path.join(out_dir, 'sessions.npy'), allow_pickle=True)
    with open(os.path.join(out_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    result = {'segments': X, 'labels': y, 'subjects': S, 'sessions': E, 'metadata': meta}
    sites_path = os.path.join(out_dir, 'sites.npy')
    if os.path.exists(sites_path):
        result['sites'] = np.load(sites_path, allow_pickle=True)
    band_path = os.path.join(out_dir, 'band_powers.npy')
    names_path = os.path.join(out_dir, 'band_names.npy')
    if os.path.exists(band_path):
        result['band_powers'] = np.load(band_path, allow_pickle=True)
    if os.path.exists(names_path):
        result['band_names'] = np.load(names_path, allow_pickle=True)
    return result


if __name__ == '__main__':
    preprocess_all(
        ucsd_root="/kaggle/input/datasets/nikhilkushwaha2529/pd-ucsd/ds002778",
        iowa_root="/kaggle/input/datasets/nikhilkushwaha2529/pd-unm-iowa/Data and Code/Dataset/IowaDataset/Raw data",
        iowa_updrs_xlsx="/kaggle/input/datasets/nikhilkushwaha2529/pd-unm-iowa/Data and Code/Dataset/IowaDataset/DataIowa.xlsx",
        unm_on_mat="/kaggle/input/datasets/nikhilkushwaha2529/pd-unm-iowa/Data and Code/Dataset/UNMDataset/Organized_data/EEG_Jim_rest_Unsegmented_WithAllChannels.mat",
        unm_off_mat="/kaggle/input/datasets/nikhilkushwaha2529/pd-unm-iowa/Data and Code/Dataset/UNMDataset_OFF/Organized_data/EEG_Jim_rest_OFF_Unsegmented_WithAllChannels.mat",
        unm_updrs_xlsx="/kaggle/input/datasets/nikhilkushwaha2529/pd-unm-iowa/Data and Code/Dataset/UNMDataset_OFF/IMPORT_ME_REST.xlsx",
        out_dir="./preprocessed",
    )

