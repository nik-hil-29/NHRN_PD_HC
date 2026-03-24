"""
preprocessing.py
================
UCSD ds002778 EEG dataset preprocessing for NHRN PD classification.

File structure expected:
  PD_DATA/
    HC/  sub-hcXX_ses-hc_task-rest_eeg.bdf
    PD/  sub-pdXX_ses-off_task-rest_eeg.bdf
         sub-pdXX_ses-on_task-rest_eeg.bdf  

Outputs (saved to ./preprocessed/):
  segments.npy  — shape (N, 40, 1024)  float32
  labels.npy    — shape (N,)           int   0=HC, 1=PD
  subjects.npy  — shape (N,)           str   subject ID per segment
  metadata.pkl  — dict with UPDRS, medication state, subject list
"""

import os
import glob
import pickle
import numpy as np
import mne
from pathlib import Path

mne.set_log_level("WARNING")

# ── constants ──────────────────────────────────────────────────────────────────
FS          = 512          # sampling frequency (Hz)
WIN_SEC     = 2            # window length (s)
WIN_SAMPLES = FS * WIN_SEC # 1024 samples per window
OVERLAP     = 0.5          # 50 % overlap → stride = 512 samples
N_CHANNELS  = 40           # UCSD uses 40 EEG channels

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


# ── helpers ────────────────────────────────────────────────────────────────────

def load_bdf(filepath: str) -> mne.io.Raw | None:
    """Load a .bdf file and return a Raw object, or None on failure."""
    try:
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
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

    # Try exact match first
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


def preprocess_raw(raw: mne.io.Raw) -> np.ndarray:
    """
    Pipeline:
      1. Re-reference to common average
      2. Bandpass filter 1–45 Hz
      3. Notch filter at 50 Hz (or 60 Hz)
      4. Resample to FS if needed
      5. Return numpy array (n_channels, n_times)
    """
    # Resample if needed
    if round(raw.info['sfreq']) != FS:
        raw.resample(FS, npad='auto')

    # Common average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Bandpass
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin',
               skip_by_annotation='edge', verbose=False)

    # Notch (50 Hz mains; change to 60 for North America)
    raw.notch_filter(freqs=[50.0], verbose=False)

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


def parse_subject_id(filepath: str) -> tuple[str, str, str]:
    """
    Returns (subject_id, session, label_str).
    E.g. 'sub-pd11_ses-off_task-rest_eeg.bdf'
       → ('sub-pd11', 'off', 'pd')
    """
    fname = Path(filepath).stem            # remove .bdf
    parts = fname.split('_')
    sub   = parts[0]                       # sub-hcXX or sub-pdXX
    ses_part = [p for p in parts if p.startswith('ses-')]
    ses   = ses_part[0].replace('ses-', '') if ses_part else 'unknown'
    label = 'hc' if 'hc' in sub.lower() else 'pd'
    return sub, ses, label


# ── main preprocessing function ───────────────────────────────────────────────

def preprocess_ucsd(data_root: str,
                    out_dir:   str = './preprocessed',
                    overlap:   float = OVERLAP) -> dict:
    """
    Process all .bdf files under data_root/HC and data_root/PD.

    Parameters
    ----------
    data_root : str
        Path to the PD_DATA directory containing HC/ and PD/ subdirectories.
    out_dir : str
        Directory to save .npy and .pkl outputs.
    overlap : float
        Window overlap fraction (0–1).

    Returns
    -------
    dict with keys: segments, labels, subjects, sessions, metadata
    """
    os.makedirs(out_dir, exist_ok=True)
    stride = int(WIN_SAMPLES * (1 - overlap))

    all_segs    = []
    all_labels  = []
    all_subjects= []
    all_sessions= []
    metadata    = {}   # subject_id → {label, session, updrs, n_segs}

    for class_dir, label_int in [('HC', 0), ('PD', 1)]:
        pattern = os.path.join(data_root, class_dir, '*.bdf')
        files   = sorted(glob.glob(pattern))
        print(f"\n[{class_dir}] Found {len(files)} files")

        for fpath in files:
            sub, ses, _ = parse_subject_id(fpath)
            print(f"  Processing {sub} ses={ses} ... ", end='', flush=True)

            raw = load_bdf(fpath)
            if raw is None:
                continue

            raw = select_channels(raw)
            if raw is None:
                print("SKIP (channel issue)")
                continue

            data = preprocess_raw(raw)

            # Pad channels to N_CHANNELS if fewer were found
            if data.shape[0] < N_CHANNELS:
                pad = np.zeros((N_CHANNELS - data.shape[0], data.shape[1]),
                               dtype=np.float32)
                data = np.concatenate([data, pad], axis=0)

            segs = segment(data, WIN_SAMPLES, stride)
            segs = z_score_normalize(segs)

            n = len(segs)
            print(f"{n} segments")

            all_segs.append(segs)
            all_labels.extend([label_int] * n)
            all_subjects.extend([sub] * n)
            all_sessions.extend([ses] * n)

            # Collect metadata
            updrs_on  = UPDRS_TABLE.get(sub, {}).get('on',  None)
            updrs_off = UPDRS_TABLE.get(sub, {}).get('off', None)
            updrs     = updrs_on if ses == 'on' else updrs_off

            if sub not in metadata:
                metadata[sub] = {
                    'label':    label_int,
                    'sessions': [],
                    'updrs_on': updrs_on,
                    'updrs_off': updrs_off,
                }
            metadata[sub]['sessions'].append({
                'session': ses,
                'updrs':   updrs,
                'n_segs':  n,
            })

    if len(all_segs) == 0:
        raise RuntimeError("No data loaded. Check data_root path.")

    X = np.concatenate(all_segs, axis=0)          # (N, 40, 1024)
    y = np.array(all_labels,  dtype=np.int32)      # (N,)
    S = np.array(all_subjects, dtype=object)        # (N,)
    E = np.array(all_sessions, dtype=object)        # (N,)

    print(f"\n── Summary ──────────────────────────────────")
    print(f"  Total segments : {len(X)}")
    print(f"  HC segments    : {(y == 0).sum()}")
    print(f"  PD segments    : {(y == 1).sum()}")
    print(f"  Unique subjects: {len(np.unique(S))}")
    print(f"  Shape          : {X.shape}  dtype={X.dtype}")

    # Save
    np.save(os.path.join(out_dir, 'segments.npy'),  X)
    np.save(os.path.join(out_dir, 'labels.npy'),    y)
    np.save(os.path.join(out_dir, 'subjects.npy'),  S)
    np.save(os.path.join(out_dir, 'sessions.npy'),  E)
    with open(os.path.join(out_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nSaved to {out_dir}/")
    return {'segments': X, 'labels': y, 'subjects': S,
            'sessions': E, 'metadata': metadata}


def load_preprocessed(out_dir: str = './preprocessed') -> dict:
    """Load previously saved preprocessed data."""
    X   = np.load(os.path.join(out_dir, 'segments.npy'),  allow_pickle=True)
    y   = np.load(os.path.join(out_dir, 'labels.npy'),    allow_pickle=True)
    S   = np.load(os.path.join(out_dir, 'subjects.npy'),  allow_pickle=True)
    E   = np.load(os.path.join(out_dir, 'sessions.npy'),  allow_pickle=True)
    with open(os.path.join(out_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return {'segments': X, 'labels': y, 'subjects': S,
            'sessions': E, 'metadata': meta}


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_ROOT = ''
    OUT_DIR   = './preprocessed'

    result = preprocess_ucsd(data_root=DATA_ROOT, out_dir=OUT_DIR)
    print("\nDone.")
