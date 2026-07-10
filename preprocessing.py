"""
preprocessing.py
================
UCSD ds002778 and Iowa/NLAB EEG dataset preprocessing for NHRN PD classification.

File structure expected:
  BIDS layout:
    ds002778/
      sub-hcXX/ses-hc/eeg/sub-hcXX_ses-hc_task-rest_eeg.bdf
      sub-pdXX/ses-off/eeg/sub-pdXX_ses-off_task-rest_eeg.bdf
      sub-pdXX/ses-on/eeg/sub-pdXX_ses-on_task-rest_eeg.bdf

  Legacy flat layout is also supported:
    PD_DATA/
      HC/sub-hcXX_ses-hc_task-rest_eeg.bdf
      PD/sub-pdXX_ses-off_task-rest_eeg.bdf

  Iowa/NLAB BrainVision layout is also supported:
    PD_IOWA_NLAB/
      Control1004.vhdr
      Control1004.eeg
      Control1004.vmrk
      PD1004.vhdr
      PD1004.eeg
      PD1004.vmrk

Outputs (saved to ./preprocessed/):
  segments.npy  — shape (N, 40, 1024)  float32
  labels.npy    — shape (N,)           int   0=HC, 1=PD
  subjects.npy  — shape (N,)           str   subject ID per segment
  sessions.npy  — shape (N,)           str   session per segment
  sites.npy     — shape (N,)           str   dataset/site per segment
  band_powers.npy — shape (N, 5)       Welch relative power per epoch
  session_band_powers.csv              subject/session-level aggregates
  metadata.pkl  — dict with UPDRS, medication state, subject list
"""

import csv
import os
import pickle
from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch

mne.set_log_level("WARNING")

# ── constants ──────────────────────────────────────────────────────────────────
FS          = 512          # sampling frequency (Hz)
WIN_SEC     = 2            # window length (s)
WIN_SAMPLES = FS * WIN_SEC # 1024 samples per window
OVERLAP     = 0.5          # 50 % overlap → stride = 512 samples
N_CHANNELS  = 40           # UCSD uses 40 EEG channels
DATA_FRACTION = 0.7       # 1.0 = use all discovered recordings
DATA_SEED     = 42         # deterministic subset selection

# Fixed artifact-inspection defaults for the base preprocessing pipeline.
ARTIFACT_PEAK_TO_PEAK_UV = 1000.0
ARTIFACT_FLAT_UV = 0.5
ARTIFACT_RMS_UV = 250.0
ARTIFACT_MAX_BAD_CHANNEL_FRACTION = 0.30

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'low_gamma': (30.0, 45.0),
}
BAND_NAMES = tuple(BANDS.keys())

# Standard 10-20 / 10-10 channel set for UCSD (40 ch)
UCSD_CHANNELS = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC5','FC1','FC2','FC6',
    'T7','C3','Cz','C4','T8',
    'CP5','CP1','CP2','CP6',
    'P7','P3','Pz','P4','P8',
    'PO7','PO3','POz','PO4','PO8',
    'O1','Oz','O2',
    'AF3','AF4','F5','F6','FC3','FC4','TP7'
]


UPDRS_TABLE = {
    'sub-pd9':  {'on': 16, 'off': 32},
    'sub-pd11': {'on': 20, 'off': 20},
    'sub-pd5':  {'on': 22, 'off': 28},
    'sub-pd26': {'on': 25, 'off': 42},
    'sub-pd16': {'on': 27, 'off': 32},
    'sub-pd17': {'on': 30, 'off': 30},
    'sub-pd19': {'on': 30, 'off': 38},
    'sub-pd14': {'on': 31, 'off': 38},
    'sub-pd3':  {'on': 33, 'off': 43},
    'sub-pd6':  {'on': 34, 'off': 40},
    'sub-pd23': {'on': 36, 'off': 44},
    'sub-pd22': {'on': 42, 'off': 58},
    'sub-pd12': {'on': 44, 'off': 47},
    'sub-pd28': {'on': 46, 'off': 47},
    'sub-pd13': {'on': 54, 'off': 49},
}


# ── artifact inspection and spectral summaries ────────────────────────────────

def inspect_and_reject_artifact_epochs(
    segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reject epochs contaminated by movement/ocular, muscle, or flat channels."""
    if len(segments) == 0:
        report = {
            'epochs_before_artifact_rejection': 0,
            'epochs_after_artifact_rejection': 0,
            'epochs_rejected_total': 0,
            'epochs_rejected_movement_ocular': 0,
            'epochs_rejected_muscle': 0,
            'epochs_rejected_flat': 0,
            'rejected_epoch_fraction': 0.0,
        }
        return segments, np.zeros((0,), dtype=bool), report

    peak_to_peak_uv = np.ptp(segments, axis=-1) * 1e6
    rms_uv = np.sqrt(np.mean(np.square(segments), axis=-1)) * 1e6

    movement_ocular_channels = peak_to_peak_uv > ARTIFACT_PEAK_TO_PEAK_UV
    muscle_channels = rms_uv > ARTIFACT_RMS_UV
    flat_channels = peak_to_peak_uv < ARTIFACT_FLAT_UV

    movement_ocular = (
        movement_ocular_channels.mean(axis=1) > ARTIFACT_MAX_BAD_CHANNEL_FRACTION
    )
    muscle = muscle_channels.mean(axis=1) > ARTIFACT_MAX_BAD_CHANNEL_FRACTION
    flat = flat_channels.mean(axis=1) > ARTIFACT_MAX_BAD_CHANNEL_FRACTION
    keep = ~(movement_ocular | muscle | flat)

    warning = None
    if not keep.any() and len(segments) > 0:
        warning = (
            "Artifact inspection marked every epoch. All epochs were retained "
            "to avoid an empty subject/session; inspect this recording manually."
        )
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


def harmonize_epoch_channels(segs: np.ndarray,
                             n_channels: int = N_CHANNELS) -> np.ndarray:
    """Pad or trim the epoch channel dimension to the model's 40-channel input."""
    if segs.shape[1] == n_channels:
        return segs
    if segs.shape[1] > n_channels:
        return segs[:, :n_channels, :]
    pad_width = n_channels - segs.shape[1]
    pad = np.zeros((segs.shape[0], pad_width, segs.shape[2]), dtype=segs.dtype)
    return np.concatenate([segs, pad], axis=1)


def welch_relative_band_power(segments: np.ndarray,
                              fs: int = FS) -> np.ndarray:
    """Return channel-averaged Welch relative power for the five EEG bands."""
    if len(segments) == 0:
        return np.empty((0, len(BAND_NAMES)), dtype=np.float32)

    nperseg = min(fs, segments.shape[-1])
    noverlap = nperseg // 2
    freqs, psd = welch(
        segments,
        fs=fs,
        window='hamming',
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )

    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.trapz(psd[..., total_mask], freqs[total_mask], axis=-1)
    total_power = np.maximum(total_power, 1e-12)

    band_features = []
    for low, high in BANDS.values():
        if high == 45.0:
            band_mask = (freqs >= low) & (freqs <= high)
        else:
            band_mask = (freqs >= low) & (freqs < high)
        band_power = np.trapz(psd[..., band_mask], freqs[band_mask], axis=-1)
        relative_power = band_power / total_power
        band_features.append(relative_power.mean(axis=1))
    return np.stack(band_features, axis=1).astype(np.float32)


# ── helpers ────────────────────────────────────────────────────────────────────

def canonical_channel_key(name: str) -> str:
    """Normalize EEG channel labels for matching across BDF naming styles."""
    clean = name.strip()
    if clean.upper().startswith('EEG '):
        clean = clean[4:]
    clean = clean.split('-')[0].split('_')[0].split('.')[0]
    return ''.join(ch for ch in clean.lower() if ch.isalnum())


def load_raw_recording(filepath: str, file_format: str) -> mne.io.Raw | None:
    """Load BDF or BrainVision header files and return a Raw object."""
    try:
        if file_format == 'bdf':
            raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
        elif file_format == 'brainvision':
            raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported recording format: {file_format}")
        return raw
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return None


def select_channels(raw: mne.io.Raw) -> mne.io.Raw | None:
    """
    Pick the 40 EEG channels. Falls back to the first 40 EEG channels
    in the file if the canonical list is not present.
    """
    available = raw.ch_names
    eeg_chs   = [ch for ch in available
                 if raw.get_channel_types([ch])[0] == 'eeg']

    # Try canonical matching first, allowing labels such as "EEG Fp1" or "Fp1-Ref".
    available_by_key = {}
    for ch in available:
        available_by_key.setdefault(canonical_channel_key(ch), ch)
    wanted_pairs = [
        (available_by_key[canonical_channel_key(ch)], ch)
        for ch in UCSD_CHANNELS
        if canonical_channel_key(ch) in available_by_key
    ]
    if len(wanted_pairs) >= N_CHANNELS:
        picks = [source for source, _ in wanted_pairs[:N_CHANNELS]]
        rename_map = {
            source: target
            for source, target in wanted_pairs[:N_CHANNELS]
            if source != target
        }
        raw.pick_channels(picks)
        if rename_map:
            raw.rename_channels(rename_map)
        return raw

    # Fall back to exact match.
    wanted = [ch for ch in UCSD_CHANNELS if ch in available]
    if len(wanted) >= N_CHANNELS:
        raw.pick_channels(wanted[:N_CHANNELS])
        return raw

    # Fall back: first N_CHANNELS EEG channels
    if len(eeg_chs) >= N_CHANNELS:
        print(f"  [INFO] Using first {N_CHANNELS} EEG channels (canonical list not found)")
        raw.pick_channels(eeg_chs[:N_CHANNELS])
        return raw

    print(f"  [WARN] Only {len(eeg_chs)} EEG channels found, need {N_CHANNELS}")
    if len(eeg_chs) > 0:
        raw.pick_channels(eeg_chs)
        return raw
    return None


def preprocess_raw(
        raw: mne.io.Raw) -> np.ndarray:
    """
    Pipeline:
      1. Select/harmonize the 40 common EEG electrodes
      2. Resample to 512 Hz if needed
      3. FIR Hamming-window bandpass filter 1-45 Hz
      4. Average re-reference
      5. Return numpy array (n_channels, n_times)
    """
    raw = select_channels(raw)
    if raw is None:
        raise RuntimeError("Unable to select EEG channels.")

    # Resample if needed
    if round(raw.info['sfreq']) != FS:
        raw.resample(FS, npad='auto')

    # Bandpass
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin',
               fir_window='hamming', skip_by_annotation='edge',
               verbose=False)

    # Common average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    data = raw.get_data()  # (n_channels, n_times)
    return data.astype(np.float32)


def segment(data: np.ndarray,
            win: int = WIN_SAMPLES,
            stride: int = WIN_SAMPLES // 2) -> np.ndarray:
    """
    Sliding-window segmentation.
    Returns array of shape (n_segs, n_channels, win).
    """
    n_ch, n_t = data.shape
    segs = []
    start = 0
    while start + win <= n_t:
        seg = data[:, start:start + win]
        segs.append(seg)
        start += stride
    if len(segs) == 0:
        return np.empty((0, n_ch, win), dtype=np.float32)
    return np.stack(segs, axis=0).astype(np.float32)


def z_score_normalize(segs: np.ndarray) -> np.ndarray:
    """
    Per-segment, per-channel z-score normalization.
    Input:  (N, C, T)
    Output: (N, C, T)
    """
    mean = segs.mean(axis=-1, keepdims=True)
    std  = segs.std(axis=-1,  keepdims=True) + 1e-8
    return (segs - mean) / std


def infer_site(path: Path) -> str:
    """Infer dataset/site label from path and filename."""
    text = str(path).lower()
    if path.suffix.lower() == '.vhdr' or 'iowa' in text or 'nlab' in text:
        return 'iowa'
    if 'ucsd' in text or 'ds002778' in text:
        return 'ucsd'
    return 'unknown'


def parse_subject_id(filepath: str) -> tuple[str, str, str]:
    """
    Returns (subject_id, session, label_str).
    E.g. 'sub-pd11/ses-off/eeg/sub-pd11_ses-off_task-rest_eeg.bdf'
       → ('sub-pd11', 'off', 'pd')
    """
    path = Path(filepath)
    stem_lower = path.stem.lower()

    if path.suffix.lower() == '.vhdr':
        if stem_lower.startswith('control'):
            return f"iowa-{path.stem}", 'rest', 'hc'
        if stem_lower.startswith('pd'):
            return f"iowa-{path.stem}", 'rest', 'pd'

    fname_parts = path.stem.split('_')
    path_parts = list(path.parts)

    sub_candidates = [
        part for part in fname_parts + path_parts
        if part.lower().startswith('sub-')
    ]
    ses_candidates = [
        part for part in fname_parts + path_parts
        if part.lower().startswith('ses-')
    ]

    sub = sub_candidates[0] if sub_candidates else path.stem
    ses = ses_candidates[0].replace('ses-', '') if ses_candidates else 'unknown'

    sub_lower = sub.lower()
    if sub_lower.startswith('sub-hc'):
        label = 'hc'
    elif sub_lower.startswith('sub-pd'):
        label = 'pd'
    else:
        raise ValueError(f"Cannot infer HC/PD label from subject id: {sub}")
    return sub, ses, label


def discover_eeg_recordings(data_roots: str | Path | list[str | Path]) -> list[dict[str, object]]:
    """Find UCSD BDF and Iowa/NLAB BrainVision header recordings."""
    if isinstance(data_roots, (str, Path)):
        roots = [Path(data_roots)]
    else:
        roots = [Path(root) for root in data_roots]

    recordings: list[dict[str, object]] = []
    skipped: list[str] = []
    seen: set[str] = set()

    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")

        bdf_candidates = sorted(root.rglob('*_task-rest_eeg.bdf'))
        if not bdf_candidates:
            bdf_candidates = sorted(root.rglob('*_eeg.bdf'))
        if not bdf_candidates:
            bdf_candidates = sorted(root.rglob('*.bdf'))

        vhdr_candidates = sorted(root.rglob('*.vhdr'))
        candidates = [(path, 'bdf') for path in bdf_candidates]
        candidates.extend((path, 'brainvision') for path in vhdr_candidates)

        for path, file_format in candidates:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                subject, session, label = parse_subject_id(str(path))
            except ValueError:
                skipped.append(str(path))
                continue
            recordings.append({
                'path': str(path),
                'format': file_format,
                'site': infer_site(path),
                'subject': subject,
                'session': session,
                'label': 0 if label == 'hc' else 1,
            })

    if skipped:
        print(f"  [WARN] Skipped {len(skipped)} EEG files with unknown subject labels")
    return sorted(recordings, key=lambda item: (str(item['site']), str(item['subject']), str(item['path'])))


def discover_bdf_files(data_root: str) -> list[tuple[str, int]]:
    """Backward-compatible UCSD BDF discovery helper."""
    return [
        (str(item['path']), int(item['label']))
        for item in discover_eeg_recordings(data_root)
        if item['format'] == 'bdf'
    ]


def normalize_data_fraction(data_fraction: float) -> float:
    """Accept either a 0-1 fraction or a 1-100 percentage."""
    value = float(data_fraction)
    if 1.0 < value <= 100.0:
        value = value / 100.0
    if not 0.0 < value <= 1.0:
        raise ValueError("data_fraction must be in (0, 1] or percent in (0, 100].")
    return value


def select_recordings_by_fraction(
    recordings: list[dict[str, object]],
    data_fraction: float = DATA_FRACTION,
    seed: int = DATA_SEED,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select a deterministic site/class-stratified subset of recordings."""
    fraction = normalize_data_fraction(data_fraction)
    report: dict[str, object] = {
        'fraction': fraction,
        'seed': int(seed),
        'total_recordings': len(recordings),
        'used_recordings': len(recordings),
        'groups': [],
    }
    if fraction >= 1.0 or len(recordings) == 0:
        return list(recordings), report

    groups: dict[tuple[str, int], list[int]] = {}
    for idx, item in enumerate(recordings):
        key = (str(item.get('site', 'unknown')), int(item.get('label', -1)))
        groups.setdefault(key, []).append(idx)

    rng = np.random.default_rng(seed)
    keep_indices: set[int] = set()
    for site_label, indices in sorted(groups.items()):
        site, label = site_label
        n_keep = max(1, int(np.ceil(len(indices) * fraction)))
        selected = rng.choice(indices, size=n_keep, replace=False)
        keep_indices.update(int(idx) for idx in selected)
        report['groups'].append({
            'site': site,
            'label': 'hc' if label == 0 else 'pd' if label == 1 else str(label),
            'total': len(indices),
            'used': n_keep,
        })

    limited = [
        item for idx, item in enumerate(recordings)
        if idx in keep_indices
    ]
    report['used_recordings'] = len(limited)
    return limited, report


def aggregate_session_band_powers(
    band_powers: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    sites: np.ndarray,
    metadata: dict,
) -> list[dict[str, object]]:
    """Aggregate epoch-level band powers to subject/session rows."""
    rows: list[dict[str, object]] = []
    for subject in sorted(np.unique(subjects.astype(str))):
        subject_mask = subjects.astype(str) == subject
        for session in sorted(np.unique(sessions[subject_mask].astype(str))):
            mask = subject_mask & (sessions.astype(str) == session)
            if not mask.any():
                continue
            label = int(np.round(labels[mask].mean()))
            row: dict[str, object] = {
                'site': str(sites[mask][0]),
                'subject': subject,
                'session': session,
                'label': label,
                'n_epochs': int(mask.sum()),
                'updrs': None,
            }
            for session_meta in metadata.get(subject, {}).get('sessions', []):
                if session_meta.get('session') == session:
                    row['updrs'] = session_meta.get('updrs')
                    break
            means = band_powers[mask].mean(axis=0)
            for band_name, value in zip(BAND_NAMES, means):
                row[f'{band_name}_relative_power'] = float(value)
            rows.append(row)
    return rows


def save_session_band_power_csv(rows: list[dict[str, object]],
                                out_dir: str) -> None:
    """Save subject/session-level band powers for UPDRS analysis."""
    if not rows:
        return
    fieldnames = [
        'site', 'subject', 'session', 'label', 'n_epochs', 'updrs',
        *[f'{band}_relative_power' for band in BAND_NAMES],
    ]
    path = os.path.join(out_dir, 'session_band_powers.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── main preprocessing function ───────────────────────────────────────────────

def preprocess_ucsd(data_root: str | Path | list[str | Path],
                    out_dir:   str = './preprocessed',
                    overlap:   float = OVERLAP,
                    data_fraction: float = DATA_FRACTION,
                    data_seed: int = DATA_SEED) -> dict:
    """
    Process UCSD .bdf and Iowa/NLAB BrainVision files.

    Parameters
    ----------
    data_root : str | Path | list[str | Path]
        One or more roots such as ds002778, PD_UCSD, PD_IOWA_NLAB, or legacy PD_DATA.
    out_dir : str
        Directory to save .npy and .pkl outputs.
    overlap : float
        Window overlap fraction (0–1).
    data_fraction : float
        Recording subset to load. Use 1.0 for all data, 0.8 for 80%, or 80.
        Selection is stratified by dataset site and HC/PD label.
    data_seed : int
        Random seed for deterministic recording subset selection.

    Returns
    -------
    dict with keys: segments, labels, subjects, sessions, band_powers, metadata
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the interval [0, 1).")

    os.makedirs(out_dir, exist_ok=True)
    stride = int(WIN_SAMPLES * (1 - overlap))

    all_segs    = []
    all_band_powers = []
    all_labels  = []
    all_subjects= []
    all_sessions= []
    all_sites   = []
    metadata    = {}   # subject_id → {label, session, updrs, n_segs}

    discovered_recordings = discover_eeg_recordings(data_root)
    if not discovered_recordings:
        raise RuntimeError(
            "No usable EEG files found. Point --ucsd-root at the UCSD ds002778 "
            "folder and --iowa-root at the Iowa/NLAB folder containing "
            ".vhdr/.eeg/.vmrk files."
        )

    recordings, selection_report = select_recordings_by_fraction(
        discovered_recordings,
        data_fraction=data_fraction,
        seed=data_seed,
    )
    metadata['_preprocessing'] = {
        'data_fraction': selection_report['fraction'],
        'data_seed': selection_report['seed'],
        'recordings_discovered': selection_report['total_recordings'],
        'recordings_used': selection_report['used_recordings'],
        'recording_subset_groups': selection_report['groups'],
        'selected_recordings': [str(item['path']) for item in recordings],
    }

    n_hc = sum(int(item['label']) == 0 for item in recordings)
    n_pd = sum(int(item['label']) == 1 for item in recordings)
    formats = {}
    for item in recordings:
        formats[str(item['format'])] = formats.get(str(item['format']), 0) + 1
    print(
        f"\n[EEG discovery] Found {len(discovered_recordings)} files; "
        f"using {len(recordings)} "
        f"({n_hc} HC, {n_pd} PD, formats={formats})"
    )
    if selection_report['fraction'] < 1.0:
        print(
            f"[Data fraction] Using {selection_report['fraction'] * 100:.1f}% "
            f"of recordings, stratified by site/class, seed={data_seed}"
        )
        for group in selection_report['groups']:
            print(
                f"  {group['site']} {group['label']}: "
                f"{group['used']}/{group['total']} recordings"
            )

    for item in recordings:
        fpath = str(item['path'])
        label_int = int(item['label'])
        sub = str(item['subject'])
        ses = str(item['session'])
        site = str(item['site'])
        file_format = str(item['format'])
        print(f"  Processing {site}:{sub} ses={ses} ... ", end='', flush=True)

        raw = load_raw_recording(fpath, file_format)
        if raw is None:
            continue

        try:
            data = preprocess_raw(raw)
        except RuntimeError:
            print("SKIP (channel issue)")
            continue

        raw_segs = segment(data, WIN_SAMPLES, stride)
        segs, _, artifact_report = inspect_and_reject_artifact_epochs(raw_segs)

        n = len(segs)
        print(
            f"{n} segments "
            f"(raw={artifact_report['epochs_before_artifact_rejection']}, "
            f"artifact_rejected={artifact_report['epochs_rejected_total']})"
        )
        if n == 0:
            continue

        segs = harmonize_epoch_channels(segs, N_CHANNELS)
        band_powers = welch_relative_band_power(segs, FS)
        norm_segs = z_score_normalize(segs)

        all_segs.append(norm_segs)
        all_band_powers.append(band_powers)
        all_labels.extend([label_int] * n)
        all_subjects.extend([sub] * n)
        all_sessions.extend([ses] * n)
        all_sites.extend([site] * n)

        # Collect metadata
        updrs_on  = UPDRS_TABLE.get(sub, {}).get('on',  None)
        updrs_off = UPDRS_TABLE.get(sub, {}).get('off', None)
        updrs     = updrs_on if ses == 'on' else updrs_off

        if sub not in metadata:
            metadata[sub] = {
                'label':    label_int,
                'site':     site,
                'sessions': [],
                'updrs_on': updrs_on,
                'updrs_off': updrs_off,
            }
        metadata[sub]['sessions'].append({
            'session': ses,
            'site': site,
            'format': file_format,
            'source_file': fpath,
            'updrs':   updrs,
            'n_segs':  n,
            'artifact_inspection': artifact_report,
        })

    if len(all_segs) == 0 or len(all_labels) == 0:
        raise RuntimeError("No usable segments loaded. Check data_root path and recordings.")

    X = np.concatenate(all_segs, axis=0)          # (N, 40, 1024)
    B = np.concatenate(all_band_powers, axis=0)   # (N, 5)
    y = np.array(all_labels,  dtype=np.int32)      # (N,)
    S = np.array(all_subjects, dtype=object)        # (N,)
    E = np.array(all_sessions, dtype=object)        # (N,)
    Z = np.array(all_sites, dtype=object)           # (N,)
    session_band_rows = aggregate_session_band_powers(B, y, S, E, Z, metadata)

    print(f"\n── Summary ──────────────────────────────────")
    print(f"  Total segments : {len(X)}")
    print(f"  HC segments    : {(y == 0).sum()}")
    print(f"  PD segments    : {(y == 1).sum()}")
    print(f"  Unique subjects: {len(np.unique(S))}")
    print(f"  Sites          : {dict(zip(*np.unique(Z, return_counts=True)))}")
    print(f"  Shape          : {X.shape}  dtype={X.dtype}")
    print(f"  Band powers    : {B.shape}  bands={BAND_NAMES}")

    # Save
    np.save(os.path.join(out_dir, 'segments.npy'),  X)
    np.save(os.path.join(out_dir, 'labels.npy'),    y)
    np.save(os.path.join(out_dir, 'subjects.npy'),  S)
    np.save(os.path.join(out_dir, 'sessions.npy'),  E)
    np.save(os.path.join(out_dir, 'sites.npy'),     Z)
    np.save(os.path.join(out_dir, 'band_powers.npy'), B)
    np.save(os.path.join(out_dir, 'band_names.npy'), np.array(BAND_NAMES, dtype=object))
    save_session_band_power_csv(session_band_rows, out_dir)
    with open(os.path.join(out_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nSaved to {out_dir}/")
    return {'segments': X, 'labels': y, 'subjects': S,
            'sessions': E, 'sites': Z, 'band_powers': B,
            'session_band_powers': session_band_rows,
            'metadata': metadata}


def preprocess_mixed(
    ucsd_root: str | Path,
    iowa_root: str | Path,
    out_dir: str | Path = './preprocessed',
    overlap: float = OVERLAP,
    data_fraction: float = DATA_FRACTION,
    data_seed: int = DATA_SEED,
) -> dict:
    """
    Process UCSD BDF and Iowa/NLAB BrainVision roots into one mixed output.

    The saved folder is directly consumable by ``train.py`` because all
    segments, labels, subjects, sessions, and site IDs are concatenated into
    shared ``.npy`` files.
    """
    return preprocess_ucsd(
        data_root=[ucsd_root, iowa_root],
        out_dir=str(out_dir),
        overlap=overlap,
        data_fraction=data_fraction,
        data_seed=data_seed,
    )


def load_preprocessed(out_dir: str = './preprocessed') -> dict:
    """Load previously saved preprocessed data."""
    X   = np.load(os.path.join(out_dir, 'segments.npy'),  allow_pickle=True)
    y   = np.load(os.path.join(out_dir, 'labels.npy'),    allow_pickle=True)
    S   = np.load(os.path.join(out_dir, 'subjects.npy'),  allow_pickle=True)
    E   = np.load(os.path.join(out_dir, 'sessions.npy'),  allow_pickle=True)
    with open(os.path.join(out_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    result = {'segments': X, 'labels': y, 'subjects': S,
              'sessions': E, 'metadata': meta}
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


# # ── entry point ───────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     DATA_ROOT = ''
#     OUT_DIR   = './preprocessed'

#     result = preprocess_ucsd(data_root=DATA_ROOT, out_dir=OUT_DIR)
#     print("\nDone.")

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # DATA_ROOT = '/kaggle/input/datasets/nikhilkushwaha2529/pd-iowa-nlab'
    # OUT_DIR   = './preprocessed'

    result = preprocess_mixed(
    ucsd_root="pd-ucsd/ds002778",
    iowa_root="pd-iowa-nlab",
    out_dir="./preprocessed"
)
    print("\nDone.")
