"""Train NHRN on preprocessed UCSD + Iowa + UNM EEG segments — 32-CHANNEL
COMMON-MONTAGE variant (see preprocessing_32.py / model_32.py). Only the 32
scalp channels common to all three sites are used as model input; each
site's additional channels (UCSD's EXG1-8, UNM's VEOG, Iowa's Fp1/Fp2
proxy) are used exclusively for ICA artifact rejection upstream, never as
model input. There is no regional-aggregate block in this variant.


The split is performed at subject level, separately within HC and PD subjects,
so windows from the same participant never appear in multiple splits.
"""

from __future__ import annotations

import pickle
import random
import time
import csv
import gc
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model_32 import binary_cross_entropy_loss, build_model, build_optimizer
from .preprocessing_32 import load_preprocessed


SEED = 42

# preprocessing_32.py outputs ONLY the 32-channel common scalp montage —
# there is no regional-aggregate block in this variant (all sites' extra
# electrodes beyond the common 32 are discarded after ICA; UCSD's EXG1-8,
# UNM's VEOG, and Iowa's Fp1/Fp2 proxy are used only as ICA artifact
# references, never as model input). REGIONAL_BLOCK_START is kept only so
# the shared EEGDataset/train_one_epoch code is identical to the 40-channel
# variant; the guard `x.shape[0] > REGIONAL_BLOCK_START` is always False
# for (32, T) input, so this mechanism is a permanent no-op here regardless
# of the augmentation_regional_block_dropout setting.
REGIONAL_BLOCK_START = 32



# Fixed site -> integer mapping for the site-adversarial auxiliary head.
SITE_TO_IDX = {'ucsd': 0, 'iowa': 1, 'unm': 2}
N_SITES = len(SITE_TO_IDX)


class EEGDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray | None = None,
        augment: bool = False,
        noise_std: float = 0.0,
        max_time_shift: int = 0,
        channel_dropout: float = 0.0,
        sites: np.ndarray | None = None,
        regional_block_dropout: float = 0.0,
    ):
        self.x = x   # kept as-is: ndarray or memmap, never copied here
        self.y = y
        self.indices = (
            np.arange(len(y)) if indices is None else np.asarray(indices)
        )
        self.augment = augment
        self.noise_std = float(noise_std)
        self.max_time_shift = int(max_time_shift)
        self.channel_dropout = float(channel_dropout)
        self.regional_block_dropout = float(regional_block_dropout)
        self._has_regional_block = None
        self._site_idx = None
        if sites is not None:
            sites_local = np.asarray(sites)[self.indices]
            site_keys = np.array([str(s).strip().lower() for s in sites_local])
            if self.regional_block_dropout > 0:
                self._has_regional_block = site_keys != 'ucsd'
            self._site_idx = np.array(
                [SITE_TO_IDX.get(s, -1) for s in site_keys], dtype=np.int64
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_idx = int(self.indices[index])
        # np.array(..., copy=True) pages in and copies only this one window,
        # whether self.x is a plain ndarray or a memmap.
        x_np = np.array(self.x[real_idx], dtype=np.float32, copy=True)
        x = torch.from_numpy(x_np)
        y = torch.tensor(float(self.y[real_idx]), dtype=torch.float32)
        if self.augment:
            if self.max_time_shift > 0:
                shift = int(torch.randint(
                    -self.max_time_shift,
                    self.max_time_shift + 1,
                    (1,),
                ).item())
                if shift:
                    x = torch.roll(x, shifts=shift, dims=-1)
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.channel_dropout > 0:
                drop = torch.rand(x.shape[0]) < self.channel_dropout
                if drop.any():
                    x[drop] = 0.0
            if (
                self._has_regional_block is not None
                and self._has_regional_block[index]
                and x.shape[0] > REGIONAL_BLOCK_START
                and torch.rand(1).item() < self.regional_block_dropout
            ):
                x[REGIONAL_BLOCK_START:] = 0.0
        if self._site_idx is not None:
            site = torch.tensor(int(self._site_idx[index]), dtype=torch.long)
            return x, y, site
        return x, y


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _subject_labels(subjects: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for subject in np.unique(subjects.astype(str)):
        mask = subjects.astype(str) == subject
        result[subject] = int(np.round(labels[mask].mean()))
    return result


def _split_subject_group(
    subjects: list[str],
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str]]:
    if len(subjects) < 3:
        raise ValueError(
            "At least 3 subjects per class are required for train/val/test splits."
        )

    shuffled = rng.permutation(subjects).tolist()
    n_subjects = len(shuffled)
    n_train = max(1, int(np.floor(n_subjects * train_frac)))
    n_val = max(1, int(np.floor(n_subjects * val_frac)))

    # Keep at least one held-out test subject in each class.
    excess = n_train + n_val - (n_subjects - 1)
    while excess > 0 and n_train > 1:
        n_train -= 1
        excess -= 1
    while excess > 0 and n_val > 1:
        n_val -= 1
        excess -= 1
    if excess > 0:
        raise ValueError(
            f"Cannot split {n_subjects} subjects with train_frac={train_frac} "
            f"and val_frac={val_frac}."
        )

    train_subjects = shuffled[:n_train]
    val_subjects = shuffled[n_train:n_train + n_val]
    test_subjects = shuffled[n_train + n_val:]
    return train_subjects, val_subjects, test_subjects


def subject_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray | None = None,
    train_frac: float = 0.65,
    val_frac: float = 0.20,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Return leakage-safe train/validation/test masks."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.")

    subjects = subjects.astype(str)
    labels = labels.astype(int)
    sites = None if sites is None else sites.astype(str)
    rng = np.random.default_rng(seed)
    labels_by_subject = _subject_labels(subjects, labels)
    sites_by_subject = {}
    if sites is not None:
        for subject in np.unique(subjects):
            values, counts = np.unique(sites[subjects == subject], return_counts=True)
            sites_by_subject[str(subject)] = str(values[np.argmax(counts)])

    class_labels = set(labels_by_subject.values())
    if class_labels != {0, 1}:
        raise ValueError("Dataset must contain both HC (0) and PD (1) subjects.")

    if sites_by_subject:
        grouped: dict[tuple[str, int], list[str]] = {}
        for subject, subject_label in labels_by_subject.items():
            grouped.setdefault(
                (sites_by_subject.get(subject, "unknown"), subject_label),
                [],
            ).append(subject)
        if any(len(subject_group) < 3 for subject_group in grouped.values()):
            grouped = {
                ("all", class_label): [
                    subject
                    for subject, subject_label in labels_by_subject.items()
                    if subject_label == class_label
                ]
                for class_label in sorted(class_labels)
            }
    else:
        grouped = {
            ("all", class_label): [
                subject
                for subject, subject_label in labels_by_subject.items()
                if subject_label == class_label
            ]
            for class_label in sorted(class_labels)
        }

    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []
    for class_subjects in grouped.values():
        tr, va, te = _split_subject_group(class_subjects, train_frac, val_frac, rng)
        train_subjects.extend(tr)
        val_subjects.extend(va)
        test_subjects.extend(te)

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    train_mask = np.array([subject in train_set for subject in subjects])
    val_mask = np.array([subject in val_set for subject in subjects])
    test_mask = np.array([subject in test_set for subject in subjects])

    return train_mask, val_mask, test_mask, {
        "train": sorted(train_set),
        "val": sorted(val_set),
        "test": sorted(test_set),
    }


def _split_subject_group_train_val(
    subjects: list[str],
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    if len(subjects) < 2:
        raise ValueError("At least 2 subjects per class are required for train/val splits.")

    shuffled = rng.permutation(subjects).tolist()
    n_subjects = len(shuffled)
    n_val = max(1, int(np.floor(n_subjects * val_frac)))
    n_val = min(n_val, n_subjects - 1)
    val_subjects = shuffled[:n_val]
    train_subjects = shuffled[n_val:]
    return train_subjects, val_subjects


def subject_train_val_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray | None = None,
    val_frac: float = 0.20,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Return leakage-safe train/validation masks when test is predefined."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")

    subjects = subjects.astype(str)
    labels = labels.astype(int)
    sites = None if sites is None else sites.astype(str)
    rng = np.random.default_rng(seed)
    labels_by_subject = _subject_labels(subjects, labels)
    class_labels = set(labels_by_subject.values())
    if class_labels != {0, 1}:
        raise ValueError("Training/validation data must contain both HC and PD subjects.")

    sites_by_subject = {}
    if sites is not None:
        for subject in np.unique(subjects):
            values, counts = np.unique(sites[subjects == subject], return_counts=True)
            sites_by_subject[str(subject)] = str(values[np.argmax(counts)])

    if sites_by_subject:
        grouped: dict[tuple[str, int], list[str]] = {}
        for subject, subject_label in labels_by_subject.items():
            grouped.setdefault(
                (sites_by_subject.get(subject, "unknown"), subject_label),
                [],
            ).append(subject)
        if any(len(subject_group) < 2 for subject_group in grouped.values()):
            grouped = {
                ("all", class_label): [
                    subject
                    for subject, subject_label in labels_by_subject.items()
                    if subject_label == class_label
                ]
                for class_label in sorted(class_labels)
            }
    else:
        grouped = {
            ("all", class_label): [
                subject
                for subject, subject_label in labels_by_subject.items()
                if subject_label == class_label
            ]
            for class_label in sorted(class_labels)
        }

    train_subjects: list[str] = []
    val_subjects: list[str] = []
    for class_subjects in grouped.values():
        tr, va = _split_subject_group_train_val(class_subjects, val_frac, rng)
        train_subjects.extend(tr)
        val_subjects.extend(va)

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    train_mask = np.array([subject in train_set for subject in subjects])
    val_mask = np.array([subject in val_set for subject in subjects])

    return train_mask, val_mask, {
        "train": sorted(train_set),
        "val": sorted(val_set),
    }


def holdout_site_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    holdout_site: str,
    val_frac: float = 0.20,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Train/val on non-holdout sites and test on all subjects from holdout_site."""
    subjects = subjects.astype(str)
    sites = sites.astype(str)
    holdout_key = str(holdout_site).lower()
    test_mask = np.array([site.lower() == holdout_key for site in sites])
    if not test_mask.any():
        raise ValueError(f"holdout_site={holdout_site!r} is not present in sites.npy.")

    development_mask = ~test_mask
    if not development_mask.any():
        raise ValueError("No training data remains after applying holdout_site.")

    dev_train_mask, dev_val_mask, dev_info = subject_train_val_split(
        subjects[development_mask],
        labels[development_mask],
        sites=sites[development_mask],
        val_frac=val_frac,
        seed=seed,
    )
    dev_indices = np.flatnonzero(development_mask)
    train_mask = np.zeros(len(subjects), dtype=bool)
    val_mask = np.zeros(len(subjects), dtype=bool)
    train_mask[dev_indices[dev_train_mask]] = True
    val_mask[dev_indices[dev_val_mask]] = True
    test_subjects = sorted(np.unique(subjects[test_mask]).tolist())

    return train_mask, val_mask, test_mask, {
        "train": dev_info["train"],
        "val": dev_info["val"],
        "test": test_subjects,
    }


def _normalize_site_filter(include_sites: str | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if include_sites is None:
        return None
    if isinstance(include_sites, str):
        return {include_sites.lower()}
    return {str(site).lower() for site in include_sites}


def _count_values(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(keys, counts)}


def make_subject_balanced_sampler(
    y: np.ndarray,
    subjects: np.ndarray,
    sites: np.ndarray | None = None,
    seed: int = SEED,
    num_samples: int | None = None,
    site_balanced: bool = False,
) -> WeightedRandomSampler:
    """Balance classes, subjects, and optionally dataset sites during sampling."""
    y = y.astype(int)
    subjects = subjects.astype(str)
    sites = None if sites is None else sites.astype(str)
    counts = np.bincount(y, minlength=2)
    if np.any(counts[:2] == 0):
        raise ValueError("Training split must contain both HC and PD segments.")

    labels_by_subject = _subject_labels(subjects, y)
    subject_counts = {
        subject: int((subjects == subject).sum())
        for subject in np.unique(subjects)
    }
    if site_balanced and sites is not None:
        site_by_subject = {}
        for subject in np.unique(subjects):
            subject_sites, site_counts = np.unique(
                sites[subjects == subject],
                return_counts=True,
            )
            site_by_subject[subject] = str(subject_sites[np.argmax(site_counts)])
        grouped_subjects: dict[tuple[str, int], list[str]] = {}
        for subject, subject_label in labels_by_subject.items():
            grouped_subjects.setdefault(
                (site_by_subject[subject], subject_label),
                [],
            ).append(subject)
    else:
        grouped_subjects = {
            ("all", class_label): [
                subject
                for subject, subject_label in labels_by_subject.items()
                if subject_label == class_label
            ]
            for class_label in [0, 1]
        }

    weights = np.zeros(len(y), dtype=np.float64)
    n_groups = max(len(grouped_subjects), 1)
    for index, (label, subject) in enumerate(zip(y, subjects)):
        if site_balanced and sites is not None:
            group_key = (site_by_subject[subject], int(label))
        else:
            group_key = ("all", int(label))
        n_group_subjects = max(len(grouped_subjects[group_key]), 1)
        weights[index] = 1.0 / (
            n_groups * n_group_subjects * subject_counts[subject]
        )
    weights = weights / weights.sum() * len(weights)

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler_size = len(weights) if num_samples is None else min(int(num_samples), len(weights))
    return WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        num_samples=sampler_size,
        replacement=True,
        generator=generator,
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray | None,
    sites: np.ndarray | None,
    batch_size: int,
    train: bool,
    num_workers: int,
    device: str,
    seed: int,
    indices: np.ndarray | None = None,
    samples_per_epoch: int | None = None,
    augmentation_noise_std: float = 0.0,
    augmentation_time_shift: int = 0,
    augmentation_channel_dropout: float = 0.0,
    augmentation_regional_block_dropout: float = 0.0,
    site_balanced_sampler: bool = False,
) -> DataLoader:
    """
    ``x``/``y``/``subjects``/``sites`` are the FULL dataset arrays (``x`` may
    be a memmap); ``indices`` selects the split this loader represents. No
    subset copy of ``x`` is created — see EEGDataset for why this matters on
    memory-constrained hosts.
    """
    dataset = EEGDataset(
        x,
        y,
        indices=indices,
        augment=train,
        noise_std=augmentation_noise_std if train else 0.0,
        max_time_shift=augmentation_time_shift if train else 0,
        channel_dropout=augmentation_channel_dropout if train else 0.0,
        sites=sites if train else None,
        regional_block_dropout=augmentation_regional_block_dropout if train else 0.0,
    )
    if train:
        if subjects is None:
            raise ValueError("Training loader requires subject IDs.")
        idx = dataset.indices
        y_local = np.asarray(y)[idx]
        subjects_local = np.asarray(subjects)[idx]
        sites_local = None if sites is None else np.asarray(sites)[idx]
        sampler = make_subject_balanced_sampler(
            y_local,
            subjects_local,
            sites=sites_local,
            seed=seed,
            num_samples=samples_per_epoch,
            site_balanced=site_balanced_sampler,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    lr_pos = sensitivity / (1.0 - specificity + 1e-9)
    lr_neg = (1.0 - sensitivity) / (specificity + 1e-9)
    dor = lr_pos / (lr_neg + 1e-9)

    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_score = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": auc_score,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "dor": dor,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def binary_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """BCE with optional mild label smoothing to reduce overconfident memorization."""
    target = target.float()
    if label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy(probabilities, target)
    return binary_cross_entropy_loss(probabilities, target)


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Standard mixup (Zhang et al., 2018): blend each window with a randomly
    paired window from the same batch. This directly targets subject-level
    memorization — the model can no longer learn an exact per-subject
    signature since no two training examples are ever seen unmixed twice.
    Returns (x_mixed, y_a, y_b, lam); combine losses as
    lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b).
    """
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


def make_site_class_weights(sites_train: np.ndarray) -> torch.Tensor:
    """
    Inverse-frequency weights for the site cross-entropy loss, indexed by
    SITE_TO_IDX. Without this, the site discriminator (and therefore the
    invariance pressure it puts on the shared backbone) is biased toward
    whichever site has the most segments — UNM here (~48% of the combined
    training set vs. Iowa's ~21%) — leaving the smallest site's real signal
    less protected than the two larger sites'.
    """
    site_keys = np.array([str(s).strip().lower() for s in sites_train])
    counts = np.array([
        max(1, (site_keys == name).sum()) for name in SITE_TO_IDX
    ], dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)  # inverse frequency, mean-normalized
    return torch.tensor(weights, dtype=torch.float32)


def set_adversarial_lambda(model: torch.nn.Module, lambda_: float) -> None:
    """Set the site-discriminator's gradient-reversal strength, handling
    an optional nn.DataParallel wrapper."""
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    if getattr(target, 'use_site_adversarial', False):
        target.site_discriminator.set_lambda(lambda_)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.10,
    mixup_alpha: float = 0.0,
    scaler: torch.amp.GradScaler | None = None,
    adversarial_lambda: float = 0.0,
    site_class_weights: torch.Tensor | None = None,
) -> tuple[float, float]:
    """
    ``adversarial_lambda > 0`` enables domain-adversarial (site-invariance)
    training: the model additionally predicts recording site from the same
    pooled feature the classifier uses, via a gradient-reversal layer, so
    the shared representation is pushed away from exploiting site identity
    as a shortcut for the PD/HC decision. Requires the loader's dataset to
    yield (x, y, site) triples (see EEGDataset) and the model to have been
    built with use_site_adversarial=True.

    ``site_class_weights``: inverse-frequency weights for the site
    cross-entropy (e.g. from make_site_class_weights). Without this, sites
    with fewer segments (e.g. Iowa, ~21% of the combined dataset vs. UNM's
    ~48%) get systematically less attention from the site classifier, so
    the invariance pressure protects the majority sites' features more
    than the minority site's — observed directly as a collapsed test AUC
    on the smallest site while the two larger sites improved.

    Returns (mean_classification_loss, train_accuracy) — the classification
    loss only (BCE, optionally +mixup), NOT including the site-adversarial
    loss term, so it stays directly comparable to evaluate_loader's reported
    validation loss (which never includes a site term).
    """
    model.train()
    total_cls_loss = 0.0
    correct = 0
    n_total = 0
    use_amp = scaler is not None and scaler.is_enabled()
    use_adversarial = adversarial_lambda > 0
    if use_adversarial:
        set_adversarial_lambda(model, adversarial_lambda)
    if site_class_weights is not None:
        site_class_weights = site_class_weights.to(device)

    for batch in loader:
        if len(batch) == 3:
            x_batch, y_batch, site_batch = batch
            site_batch = site_batch.to(device, non_blocking=True)
        else:
            x_batch, y_batch = batch
            site_batch = None
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        request_site = use_adversarial and site_batch is not None and mixup_alpha == 0
        with torch.amp.autocast('cuda', enabled=use_amp):
            if mixup_alpha > 0:
                x_mixed, y_a, y_b, lam = mixup_batch(x_batch, y_batch, mixup_alpha)
                probabilities = model(x_mixed)
                site_logits = None
            elif request_site:
                probabilities, site_logits = model(x_batch, return_site_logits=True)
            else:
                probabilities = model(x_batch)
                site_logits = None
        # binary_cross_entropy on sigmoid probabilities (not logits) is not
        # autocast-safe in fp16 (can produce NaN/Inf near 0/1) — compute the
        # loss in float32 outside the autocast context. The forward pass
        # above still ran under autocast, so this doesn't lose the speed/
        # memory benefit, it just keeps the loss numerically stable.
        probabilities = probabilities.float()
        if mixup_alpha > 0:
            cls_loss = lam * binary_loss(probabilities, y_a, label_smoothing) + \
                       (1.0 - lam) * binary_loss(probabilities, y_b, label_smoothing)
            # Train accuracy reported against the dominant label (lam >= 0.5
            # convention); this is a diagnostic readout, not used for any
            # decision, so the mixup ambiguity here doesn't matter.
            acc_target = y_a if lam >= 0.5 else y_b
        else:
            cls_loss = binary_loss(probabilities, y_batch, label_smoothing)
            acc_target = y_batch

        loss = cls_loss
        if site_logits is not None:
            valid = site_batch >= 0  # guard against any unmapped site string
            if valid.any():
                site_loss = F.cross_entropy(
                    site_logits.float()[valid], site_batch[valid],
                    weight=site_class_weights,
                )
                loss = loss + site_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = len(y_batch)
        total_cls_loss += cls_loss.item() * batch_size
        correct += ((probabilities >= 0.5).float() == acc_target).sum().item()
        n_total += batch_size

    return total_cls_loss / n_total, correct / n_total * 100.0


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    n_total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            prob = model(x_batch)
        prob = prob.float()  # see train_one_epoch: BCE on probabilities is not fp16-safe
        loss = binary_cross_entropy_loss(prob, y_batch)

        batch_size = len(y_batch)
        total_loss += loss.item() * batch_size
        n_total += batch_size
        probabilities.append(prob.cpu().numpy())
        predictions.append((prob >= 0.5).float().cpu().numpy())
        labels.append(y_batch.cpu().numpy())

    if n_total == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    probs = np.concatenate(probabilities)
    preds = np.concatenate(predictions).astype(int)
    trues = np.concatenate(labels).astype(int)
    return total_loss / n_total, preds, probs, trues


def aggregate_subject_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average epoch probabilities per subject for subject-level evaluation."""
    subjects = subjects.astype(str)
    subject_ids = np.unique(subjects)
    subject_true = []
    subject_prob = []
    for subject in subject_ids:
        mask = subjects == subject
        subject_true.append(int(np.round(y_true[mask].mean())))
        subject_prob.append(float(y_prob[mask].mean()))
    subject_true_arr = np.asarray(subject_true, dtype=int)
    subject_prob_arr = np.asarray(subject_prob, dtype=np.float32)
    subject_pred_arr = (subject_prob_arr >= 0.5).astype(int)
    return subject_true_arr, subject_pred_arr, subject_prob_arr, subject_ids


def sitewise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subjects: np.ndarray,
    sites: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    """Compute segment-level and subject-level metrics separately by dataset site."""
    if sites is None:
        return {}

    result: dict[str, dict[str, Any]] = {}
    sites = sites.astype(str)
    for site in sorted(np.unique(sites)):
        mask = sites == site
        if not mask.any():
            continue
        subject_true, subject_pred, subject_prob, subject_ids = (
            aggregate_subject_predictions(
                y_true[mask],
                y_prob[mask],
                subjects[mask],
            )
        )
        result[site] = {
            "n_segments": int(mask.sum()),
            "n_subjects": int(len(subject_ids)),
            "segment": compute_metrics(y_true[mask], y_pred[mask], y_prob[mask]),
            "subject": compute_metrics(subject_true, subject_pred, subject_prob),
        }
    return result


def print_sitewise_metrics(split_name: str, metrics: dict[str, dict[str, Any]]) -> None:
    """Print compact per-site metrics for domain-shift diagnosis."""
    if not metrics:
        return

    print(f"\n  {split_name.upper()} SITE METRICS")
    print("  Site       Segs  Subj  Seg Bal  Subj Bal  Subj Acc  AUC")
    print("  --------------------------------------------------------")
    for site, values in metrics.items():
        segment = values["segment"]
        subject = values["subject"]
        print(
            f"  {site:<8} "
            f"{values['n_segments']:>6} "
            f"{values['n_subjects']:>5} "
            f"{segment['balanced_accuracy']:>7.2f}% "
            f"{subject['balanced_accuracy']:>8.2f}% "
            f"{subject['accuracy']:>8.2f}% "
            f"{segment['auc_roc']:>5.3f}"
        )


def _print_split_summary(
    split_info: dict[str, list[str]],
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    print(f"\n  Train : {n_train:>5} segs  subjects={split_info['train']}")
    print(f"  Val   : {n_val:>5} segs  subjects={split_info['val']}")
    print(f"  Test  : {n_test:>5} segs  subjects={split_info['test']}")


def train(
    data_dir: str | Path = "./preprocessed_32",
    save_dir: str | Path = "./results_32",
    epochs: int = 100,
    patience: int = 12,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 2e-3,
    dropout: float = 0.5,
    feat_dropout: float = 0.1,
    use_memory_gru: bool = False,
    use_site_adversarial: bool = False,
    adversarial_lambda_max: float = 0.3,
    mixup_alpha: float = 0.0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    device: str = "auto",
    seed: int = SEED,
    num_workers: int = 0,
    use_amp: bool = True,
    multi_gpu: bool = True,
    label_smoothing: float = 0.10,
    samples_per_epoch: int | None = 8192,
    augmentation_noise_std: float = 0.03,
    augmentation_time_shift: int = 64,
    augmentation_channel_dropout: float = 0.05,
    augmentation_regional_block_dropout: float = 0.0,  # inert for 32-channel data (no aggregate block)
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    min_epochs: int = 5,
    overfit_gap: float = 25.0,
    overfit_patience: int = 3,
    include_sites: str | list[str] | tuple[str, ...] | None = None,
    holdout_site: str | None = None,
    pretrain_site: str | None = None,
    pretrain_epochs: int = 0,
    pretrain_lr: float | None = None,
    pretrain_samples_per_epoch: int | None = None,
    site_balanced_sampler: bool = True,
    mmap_segments: bool = True,
) -> dict[str, Any]:
    """
    Train NHRN and save the best validation-loss checkpoint.

    ``mmap_segments=True`` (default) memory-maps segments.npy instead of
    loading the full combined UCSD+UNM+Iowa array into RAM, and train/val/
    test/pretrain splits are represented as index arrays rather than
    materialized copies — important on memory-constrained hosts (e.g.
    Kaggle's ~30 GiB RAM) where loading the full array plus split copies
    can exhaust available memory. Set to False only if segments.npy lives
    on slow/network storage where repeated small reads would be slower
    than one big eager load.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    if samples_per_epoch is not None and samples_per_epoch <= 0:
        samples_per_epoch = None
    if samples_per_epoch is not None and samples_per_epoch < batch_size:
        raise ValueError("samples_per_epoch must be at least batch_size, or None.")
    if augmentation_noise_std < 0:
        raise ValueError("augmentation_noise_std must be non-negative.")
    if augmentation_time_shift < 0:
        raise ValueError("augmentation_time_shift must be non-negative.")
    if not 0.0 <= augmentation_channel_dropout < 1.0:
        raise ValueError("augmentation_channel_dropout must be in [0, 1).")
    if lr_patience < 1:
        raise ValueError("lr_patience must be at least 1.")
    if not 0.0 < lr_factor < 1.0:
        raise ValueError("lr_factor must be between 0 and 1.")
    if min_epochs < 1:
        raise ValueError("min_epochs must be at least 1.")
    if overfit_patience < 1:
        raise ValueError("overfit_patience must be at least 1.")
    if pretrain_epochs < 0:
        raise ValueError("pretrain_epochs must be non-negative.")
    if pretrain_samples_per_epoch is not None and pretrain_samples_per_epoch <= 0:
        pretrain_samples_per_epoch = None
    if pretrain_lr is not None and pretrain_lr <= 0:
        raise ValueError("pretrain_lr must be positive when supplied.")

    set_seed(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print("  NHRN Training - mixed EEG dataset")
    print(f"{'=' * 60}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {epochs}  (patience={patience})")
    print(f"  Batch size  : {batch_size}")
    print(f"  Samples/ep  : {samples_per_epoch if samples_per_epoch else 'all'}")
    print(f"  Site sampler: {'site/class/subject' if site_balanced_sampler else 'class/subject'}")
    print(f"  LR          : {lr}   weight_decay={weight_decay}")
    print(f"  Dropout     : head/fusion={dropout}  backbone={feat_dropout}")
    print(f"  Memory module: {'GRU (paper default)' if use_memory_gru else 'fast non-recurrent fallback (ablation)'}")
    print(f"  Site-adversarial: {'on, lambda_max=' + str(adversarial_lambda_max) if use_site_adversarial else 'off'}")
    print(f"  Mixup       : alpha={mixup_alpha}")
    print(f"  Label smooth: {label_smoothing}")
    print(
        "  Augment     : "
        f"noise={augmentation_noise_std}, shift={augmentation_time_shift}, "
        f"ch_drop={augmentation_channel_dropout}, "
        f"regional_block_drop={augmentation_regional_block_dropout} (UNM/Iowa only)"
    )
    print(
        f"  Overfit stop: gap>{overfit_gap:.1f} for {overfit_patience} epochs "
        f"after epoch {min_epochs}"
    )
    print(
        f"  Split       : train={train_frac:.0%}  val={val_frac:.0%}  "
        f"test={1 - train_frac - val_frac:.0%}"
    )

    data = load_preprocessed(str(data_dir), mmap=mmap_segments)
    x = data["segments"].astype(np.float32, copy=False)
    y = data["labels"].astype(int, copy=False)
    subjects = data["subjects"].astype(str)
    sites = data.get("sites")
    sites = None if sites is None else sites.astype(str)

    wanted_sites = _normalize_site_filter(include_sites)
    if wanted_sites is not None:
        if sites is None:
            raise ValueError("include_sites requires sites.npy in the preprocessed folder.")
        site_mask = np.array([site.lower() in wanted_sites for site in sites])
        if not site_mask.any():
            raise ValueError(f"No segments matched include_sites={sorted(wanted_sites)}.")
        # Materializes only the filtered subset (e.g. UCSD-only), not the
        # full combined array, so this is safe even when x is a memmap.
        x = x[site_mask]
        y = y[site_mask]
        subjects = subjects[site_mask]
        sites = sites[site_mask]

    if x.ndim != 3 or x.shape[1:] != (32, 1024):
        raise ValueError(f"Expected segments shape (N, 32, 1024), found {x.shape}.")

    print(f"\n  Data loaded : X={x.shape}  y={y.shape}")
    print(f"  HC segs     : {(y == 0).sum()}    PD segs: {(y == 1).sum()}")
    print(f"  Subjects    : {len(np.unique(subjects))}")
    if sites is not None:
        print(f"  Sites       : {_count_values(sites)}")
    if wanted_sites is not None:
        print(f"  Site filter : {sorted(wanted_sites)}")
    if holdout_site is not None:
        print(f"  Holdout site: {holdout_site} (all subjects used as test)")
    if pretrain_site and pretrain_epochs > 0:
        print(
            f"  Curriculum  : pretrain on {pretrain_site} for "
            f"{pretrain_epochs} epochs, then mixed fine-tune"
        )

    if holdout_site is not None:
        if sites is None:
            raise ValueError("holdout_site requires sites.npy in the preprocessed folder.")
        train_mask, val_mask, test_mask, split_info = holdout_site_split(
            subjects,
            y,
            sites,
            holdout_site=holdout_site,
            val_frac=val_frac,
            seed=seed,
        )
    else:
        train_mask, val_mask, test_mask, split_info = subject_split(
            subjects,
            y,
            sites=sites,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )
    # Index arrays only (no x[mask] copy): x may be a memmap over the full
    # combined dataset, and materializing train/val/test subset copies on
    # top of it is what exhausts RAM on memory-constrained hosts. Each
    # EEGDataset reads its rows lazily via these indices instead.
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    subjects_train = subjects[train_idx]
    subjects_val = subjects[val_idx]
    subjects_test = subjects[test_idx]
    y_train = y[train_idx]   # small (labels only); safe to slice eagerly
    sites_train = None if sites is None else sites[train_idx]
    sites_val = None if sites is None else sites[val_idx]
    sites_test = None if sites is None else sites[test_idx]
    _print_split_summary(split_info, len(train_idx), len(val_idx), len(test_idx))
    if sites is not None:
        print(f"  Train sites: {_count_values(sites_train)}")
        print(f"  Val sites  : {_count_values(sites_val)}")
        print(f"  Test sites : {_count_values(sites_test)}")

    train_loader = make_loader(
        x,
        y,
        subjects,
        sites,
        batch_size,
        True,
        num_workers,
        device,
        seed,
        indices=train_idx,
        samples_per_epoch=samples_per_epoch,
        augmentation_noise_std=augmentation_noise_std,
        augmentation_time_shift=augmentation_time_shift,
        augmentation_channel_dropout=augmentation_channel_dropout,
        augmentation_regional_block_dropout=augmentation_regional_block_dropout,
        site_balanced_sampler=site_balanced_sampler,
    )
    val_loader = make_loader(
        x, y, None, None, batch_size, False, num_workers, device, seed,
        indices=val_idx,
    )
    test_loader = make_loader(
        x, y, None, None, batch_size, False, num_workers, device, seed,
        indices=test_idx,
    )

    model = build_model(device=device, dropout=dropout, feat_dropout=feat_dropout,
                         use_memory_gru=use_memory_gru,
                         use_site_adversarial=use_site_adversarial, n_sites=N_SITES)

    n_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0
    if multi_gpu and n_gpus > 1:
        print(f"  Multi-GPU   : using {n_gpus} GPUs via nn.DataParallel")
        model = torch.nn.DataParallel(model)

    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    print(f"  Mixed prec. : {'enabled' if amp_enabled else 'disabled'}")


    pretrain_history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_subject_balanced_acc": [],
    }
    if pretrain_site and pretrain_epochs > 0:
        if sites_train is None:
            print("  [WARN] Iowa pretrain skipped because sites.npy is unavailable.")
        else:
            pretrain_key = str(pretrain_site).lower()
            pretrain_mask = np.array([
                str(site).lower() == pretrain_key
                for site in sites_train
            ])
            pretrain_labels = y_train[pretrain_mask]
            if not pretrain_mask.any():
                print(f"  [WARN] Pretrain skipped: no train segments for site={pretrain_site}.")
            elif len(np.unique(pretrain_labels)) < 2:
                print(
                    f"  [WARN] Pretrain skipped: site={pretrain_site} "
                    "does not contain both classes in the train split."
                )
            else:
                pretrain_budget = (
                    pretrain_samples_per_epoch
                    if pretrain_samples_per_epoch is not None
                    else samples_per_epoch
                )
                pretrain_idx = train_idx[pretrain_mask]
                pretrain_loader = make_loader(
                    x,
                    y,
                    subjects,
                    sites,
                    batch_size,
                    True,
                    num_workers,
                    device,
                    seed,
                    indices=pretrain_idx,
                    samples_per_epoch=pretrain_budget,
                    augmentation_noise_std=augmentation_noise_std,
                    augmentation_time_shift=augmentation_time_shift,
                    augmentation_channel_dropout=augmentation_channel_dropout,
                    augmentation_regional_block_dropout=augmentation_regional_block_dropout,
                    site_balanced_sampler=False,
                )
                pre_optimizer = build_optimizer(
                    model,
                    lr=pretrain_lr if pretrain_lr is not None else lr,
                    weight_decay=weight_decay,
                )
                print(
                    f"\n  Curriculum pretrain: site={pretrain_site}, "
                    f"segments={int(pretrain_mask.sum())}, "
                    f"samples/ep={pretrain_budget if pretrain_budget else 'all'}"
                )
                print(
                    f"  {'PreEp':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
                    f"{'Va Loss':>8}  {'Va Bal':>7}"
                )
                print(f"  {'-' * 44}")
                for pre_epoch in range(1, pretrain_epochs + 1):
                    pre_loss, pre_acc = train_one_epoch(
                        model,
                        pretrain_loader,
                        pre_optimizer,
                        device,
                        label_smoothing=label_smoothing,
                        mixup_alpha=mixup_alpha,
                        scaler=scaler,
                    )
                    pre_val_loss, pre_val_preds, pre_val_probs, pre_val_true = (
                        evaluate_loader(model, val_loader, device, use_amp=amp_enabled)
                    )
                    pre_val_subject_true, pre_val_subject_preds, pre_val_subject_probs, _ = (
                        aggregate_subject_predictions(
                            pre_val_true,
                            pre_val_probs,
                            subjects_val,
                        )
                    )
                    pre_val_metrics = compute_metrics(
                        pre_val_subject_true,
                        pre_val_subject_preds,
                        pre_val_subject_probs,
                    )
                    pre_val_bal = float(pre_val_metrics["balanced_accuracy"])
                    pretrain_history["train_loss"].append(pre_loss)
                    pretrain_history["train_acc"].append(pre_acc)
                    pretrain_history["val_loss"].append(pre_val_loss)
                    pretrain_history["val_subject_balanced_acc"].append(pre_val_bal)
                    if pre_epoch == 1 or pre_epoch == pretrain_epochs or pre_epoch % 2 == 0:
                        print(
                            f"  {pre_epoch:>5}  {pre_loss:>8.4f}  "
                            f"{pre_acc:>6.2f}%  {pre_val_loss:>8.4f}  "
                            f"{pre_val_bal:>6.2f}%"
                        )
                # Free the pretrain loader/optimizer and any cached CUDA
                # blocks before the (larger, multi-site) main phase starts.
                del pretrain_loader, pre_optimizer
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr / 50.0,
    )

    param_source = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_params, trainable_params = param_source.count_parameters()
    print(f"\n  Model params: {total_params:,} ({trainable_params:,} trainable)")
    print(f"{'-' * 60}")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_subject_acc": [],
        "val_subject_balanced_acc": [],
        "generalization_gap": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_subject_acc = 0.0
    best_val_subject_balanced_acc = -1.0
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0
    overfit_count = 0
    start_time = time.time()

    site_class_weights = (
        make_site_class_weights(sites_train)
        if use_site_adversarial and sites_train is not None
        else None
    )
    if site_class_weights is not None:
        weight_str = ", ".join(
            f"{name}={w:.2f}" for name, w in zip(SITE_TO_IDX, site_class_weights.tolist())
        )
        print(f"  Site CE weights: {weight_str} (inverse-frequency)")

    print(
        f"\n  {'Ep':>4}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
        f"{'Va Loss':>8}  {'Va Seg':>7}  {'Va Subj':>7}  "
        f"{'Va Bal':>7}  {'Gap':>7}  {'LR':>8}  {'Best':>7}"
    )
    print(f"  {'-' * 60}")

    for epoch in range(1, epochs + 1):
        if use_site_adversarial:
            # Standard DANN ramp: lambda goes 0 -> adversarial_lambda_max
            # smoothly over training (progress in [0, 1]), so early epochs
            # focus on the main PD/HC task before the site-invariance
            # pressure ramps up.
            progress = min(1.0, (epoch - 1) / max(1, epochs - 1))
            current_lambda = adversarial_lambda_max * (2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)
        else:
            current_lambda = 0.0

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            scaler=scaler,
            adversarial_lambda=current_lambda,
            site_class_weights=site_class_weights,
        )
        val_loss, val_preds, val_probs, val_true = evaluate_loader(
            model, val_loader, device, use_amp=amp_enabled
        )
        val_acc = accuracy_score(val_true, val_preds) * 100.0
        (
            val_subject_true,
            val_subject_preds,
            val_subject_probs,
            _,
        ) = aggregate_subject_predictions(val_true, val_probs, subjects_val)
        val_subject_metrics = compute_metrics(
            val_subject_true,
            val_subject_preds,
            val_subject_probs,
        )
        val_subject_acc = float(val_subject_metrics["accuracy"])
        val_subject_balanced_acc = float(val_subject_metrics["balanced_accuracy"])
        generalization_gap = train_acc - val_subject_balanced_acc

        scheduler.step(val_subject_balanced_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_subject_acc"].append(val_subject_acc)
        history["val_subject_balanced_acc"].append(val_subject_balanced_acc)
        history["generalization_gap"].append(generalization_gap)
        history["lr"].append(current_lr)

        improved = (
            val_subject_balanced_acc > best_val_subject_balanced_acc + 1e-8
            or (
                abs(val_subject_balanced_acc - best_val_subject_balanced_acc) <= 1e-8
                and val_loss < best_val_loss - 1e-5
            )
        )
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_subject_acc = val_subject_acc
            best_val_subject_balanced_acc = val_subject_balanced_acc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            flag = "*"
        else:
            no_improve += 1
            flag = ""

        if (
            epoch >= min_epochs
            and not improved
            and generalization_gap >= overfit_gap
        ):
            overfit_count += 1
        else:
            overfit_count = 0

        if epoch == 1 or improved or epoch % 5 == 0:
            print(
                f"  {epoch:>4}  {train_loss:>8.4f}  {train_acc:>6.2f}%  "
                f"{val_loss:>8.4f}  {val_acc:>6.2f}%  "
                f"{val_subject_acc:>6.2f}%  "
                f"{val_subject_balanced_acc:>6.2f}%  "
                f"{generalization_gap:>6.2f}%  "
                f"{current_lr:>8.6f}  {flag:>7}"
            )

        if overfit_count >= overfit_patience:
            print(
                f"\n  Overfit guard stopped at epoch {epoch}: "
                f"train acc is {generalization_gap:.2f} points above "
                f"validation subject balanced accuracy for "
                f"{overfit_patience} epochs."
            )
            break

        if no_improve >= patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")
    print(
        f"  Best epoch    : {best_epoch}  "
        f"val loss={best_val_loss:.4f}  "
        f"val seg acc={best_val_acc:.2f}%  "
        f"val subject acc={best_val_subject_acc:.2f}%  "
        f"val subject bal={best_val_subject_balanced_acc:.2f}%"
    )

    model.load_state_dict(best_state)
    best_val_loss_eval, best_val_preds, best_val_probs, best_val_true = evaluate_loader(
        model,
        val_loader,
        device,
        use_amp=amp_enabled,
    )
    best_val_site_metrics = sitewise_metrics(
        best_val_true,
        best_val_preds,
        best_val_probs,
        subjects_val,
        sites_val,
    )
    print_sitewise_metrics("validation", best_val_site_metrics)

    test_loss, test_preds, test_probs, test_true = evaluate_loader(
        model,
        test_loader,
        device,
        use_amp=amp_enabled,
    )
    test_metrics = compute_metrics(test_true, test_preds, test_probs)
    test_subject_true, test_subject_preds, test_subject_probs, test_subject_ids = (
        aggregate_subject_predictions(test_true, test_probs, subjects_test)
    )
    test_subject_metrics = compute_metrics(
        test_subject_true,
        test_subject_preds,
        test_subject_probs,
    )
    test_site_metrics = sitewise_metrics(
        test_true,
        test_preds,
        test_probs,
        subjects_test,
        sites_test,
    )

    print(f"\n{'=' * 60}")
    print("  TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"  Loss        : {test_loss:.4f}")
    print(f"  Accuracy    : {test_metrics['accuracy']:.2f}%")
    print(f"  Balanced Acc: {test_metrics['balanced_accuracy']:.2f}%")
    print(f"  F1-Score    : {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC     : {test_metrics['auc_roc']:.4f}")
    print(f"  Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity : {test_metrics['specificity']:.4f}")
    print(f"  PPV         : {test_metrics['ppv']:.4f}")
    print(f"  NPV         : {test_metrics['npv']:.4f}")
    print(f"  LR+         : {test_metrics['lr_pos']:.2f}")
    print(f"  LR-         : {test_metrics['lr_neg']:.4f}")
    print(f"  DOR         : {test_metrics['dor']:.1f}")
    print(f"  Cohen kappa : {test_metrics['kappa']:.4f}")
    print(f"  MCC         : {test_metrics['mcc']:.4f}")
    print(f"\n  Subject Accuracy    : {test_subject_metrics['accuracy']:.2f}%")
    print(f"  Subject Balanced Acc: {test_subject_metrics['balanced_accuracy']:.2f}%")
    print(f"  Subject F1-Score    : {test_subject_metrics['f1']:.4f}")
    print_sitewise_metrics("test", test_site_metrics)

    cm = confusion_matrix(test_true, test_preds, labels=[0, 1])
    print("\n  Confusion Matrix:")
    print("              Pred HC   Pred PD")
    print(f"  True HC  :  {cm[0, 0]:>7}   {cm[0, 1]:>7}")
    print(f"  True PD  :  {cm[1, 0]:>7}   {cm[1, 1]:>7}")
    print(f"{'=' * 60}")

    model_path = save_path / "nhrn_best.pt"
    # Save with plain (non-"module."-prefixed) keys regardless of whether
    # training used nn.DataParallel, so the checkpoint loads directly into
    # a plain NHRN instance later (build_model(...).load_state_dict(...)).
    portable_state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in best_state.items()
    }
    torch.save(portable_state, model_path)
    print(f"\n  Model state saved -> {model_path}")

    results = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_subject_metrics": test_subject_metrics,
        "history": history,
        "split_info": split_info,
        "test_predictions": test_preds,
        "test_probabilities": test_probs,
        "test_labels": test_true,
        "test_subject_ids": test_subject_ids,
        "test_subject_predictions": test_subject_preds,
        "test_subject_probabilities": test_subject_probs,
        "test_subject_labels": test_subject_true,
        "best_val_loss_eval": best_val_loss_eval,
        "best_val_site_metrics": best_val_site_metrics,
        "test_site_metrics": test_site_metrics,
        "pretrain_history": pretrain_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_subject_acc": best_val_subject_acc,
        "best_val_subject_balanced_acc": best_val_subject_balanced_acc,
        "elapsed_s": elapsed,
        "hparams": {
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "feat_dropout": feat_dropout,
            "use_memory_gru": use_memory_gru,
            "use_site_adversarial": use_site_adversarial,
            "adversarial_lambda_max": adversarial_lambda_max,
            "use_amp": use_amp,
            "multi_gpu": multi_gpu,
            "mixup_alpha": mixup_alpha,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
            "label_smoothing": label_smoothing,
            "samples_per_epoch": samples_per_epoch,
            "augmentation_noise_std": augmentation_noise_std,
            "augmentation_time_shift": augmentation_time_shift,
            "augmentation_channel_dropout": augmentation_channel_dropout,
            "augmentation_regional_block_dropout": augmentation_regional_block_dropout,
            "lr_factor": lr_factor,
            "min_epochs": min_epochs,
            "overfit_gap": overfit_gap,
            "overfit_patience": overfit_patience,
            "include_sites": include_sites,
            "holdout_site": holdout_site,
            "pretrain_site": pretrain_site,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_lr": pretrain_lr,
            "pretrain_samples_per_epoch": pretrain_samples_per_epoch,
            "site_balanced_sampler": site_balanced_sampler,
            "mmap_segments": mmap_segments,
        },
    }

    results_path = save_path / "train_results.pkl"
    with results_path.open("wb") as handle:
        pickle.dump(results, handle)
    print(f"  Results saved -> {results_path}")

    _plot_training_curves(history, save_path)
    _plot_confusion_matrix(test_true, test_preds, test_metrics["accuracy"], save_path)
    _plot_roc_curve(test_true, test_probs, test_metrics["auc_roc"], save_path)
    return results


def _plot_training_curves(history: dict[str, list[float]], save_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", lw=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Segment Acc", lw=2, linestyle="--")
    if "val_subject_acc" in history:
        axes[1].plot(
            epochs,
            history["val_subject_acc"],
            label="Val Subject Acc",
            lw=2,
            linestyle=":",
        )
    if "val_subject_balanced_acc" in history:
        axes[1].plot(
            epochs,
            history["val_subject_balanced_acc"],
            label="Val Subject Bal Acc",
            lw=2,
            linestyle="-.",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["lr"], lw=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = save_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved -> {path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float | int,
    save_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(image, ax=ax)

    classes = ["HC", "PD"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - Test Set\nAccuracy: {accuracy:.2f}%")

    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100.0 if cm[i].sum() else 0.0
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    path = save_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_value: float | int,
    save_dir: Path,
) -> None:
    if len(np.unique(y_true)) < 2 or np.isnan(float(auc_value)):
        print("  ROC curve skipped because the test labels contain one class.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - NHRN Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = save_dir / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved -> {path}")


DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "data_dir": "./preprocessed_32",
    "save_dir": "./results_32",
    "device": "cuda",
    "epochs": 500,
    "patience": 12,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 2e-3,
    "dropout": 0.5,
    "feat_dropout": 0.1,
    "use_memory_gru": False,
    "use_site_adversarial": False,
    "adversarial_lambda_max": 0.3,
    "use_amp": True,
    "multi_gpu": True,
    "mixup_alpha": 0.0,
    "label_smoothing": 0.10,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "samples_per_epoch": 8192,
    "augmentation_noise_std": 0.03,
    "augmentation_time_shift": 64,
    "augmentation_channel_dropout": 0.05,
    "augmentation_regional_block_dropout": 0.0,
    "lr_patience": 2,
    "lr_factor": 0.5,
    "min_epochs": 5,
    "overfit_gap": 25.0,
    "overfit_patience": 3,
    "pretrain_site": "ucsd",
    "pretrain_epochs": 0,
    "pretrain_lr": 1e-4,
    "pretrain_samples_per_epoch": 8192,
    "site_balanced_sampler": True,
    "mmap_segments": True,
    "seed": SEED,
    "num_workers": 0,
}


def _domain_shift_summary_row(scenario: str, result: dict[str, Any]) -> dict[str, Any]:
    test_metrics = result["test_metrics"]
    test_subject_metrics = result["test_subject_metrics"]
    return {
        "scenario": scenario,
        "best_epoch": result["best_epoch"],
        "best_val_subject_balanced_acc": result["best_val_subject_balanced_acc"],
        "test_segment_acc": test_metrics["accuracy"],
        "test_segment_balanced_acc": test_metrics["balanced_accuracy"],
        "test_subject_acc": test_subject_metrics["accuracy"],
        "test_subject_balanced_acc": test_subject_metrics["balanced_accuracy"],
        "test_auc_roc": test_metrics["auc_roc"],
        "test_f1": test_metrics["f1"],
    }


def run_domain_shift_diagnostics(
    data_dir: str | Path = "./preprocessed_32",
    save_root: str | Path = "./results_32_domain_shift",
    base_config: dict[str, Any] | None = None,
    scenarios: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Run mixed, single-site, and cross-site experiments to diagnose domain shift.

    The important comparison is:
      - UCSD-only and Iowa-only high, cross-site low: domain shift/channel mismatch.
      - all scenarios low: preprocessing, labels, or model training still need work.
      - mixed high and cross-site high: architecture/training is adequate.
    """
    data_path = Path(data_dir)
    sites_path = data_path / "sites.npy"
    if not sites_path.exists():
        raise FileNotFoundError(
            "Domain-shift diagnostics require sites.npy. Re-run preprocessing "
            "with mixed UCSD/Iowa support."
        )

    available_sites = sorted(np.unique(np.load(sites_path, allow_pickle=True).astype(str)))
    available_site_keys = {site.lower(): site for site in available_sites}
    print(f"\nAvailable sites for diagnostics: {available_sites}")

    if scenarios is None:
        scenarios = [{"name": "mixed_subject_split"}]
        if "ucsd" in available_site_keys:
            scenarios.append({"name": "ucsd_only", "include_sites": ["ucsd"]})
        if "iowa" in available_site_keys:
            scenarios.append({"name": "iowa_only", "include_sites": ["iowa"]})
        if "ucsd" in available_site_keys and "iowa" in available_site_keys:
            scenarios.extend([
                {
                    "name": "train_ucsd_test_iowa",
                    "include_sites": ["ucsd", "iowa"],
                    "holdout_site": "iowa",
                },
                {
                    "name": "train_iowa_test_ucsd",
                    "include_sites": ["ucsd", "iowa"],
                    "holdout_site": "ucsd",
                },
            ])

    save_root_path = Path(save_root)
    save_root_path.mkdir(parents=True, exist_ok=True)
    config = dict(DEFAULT_TRAIN_CONFIG)
    if base_config:
        config.update(base_config)
    config["data_dir"] = str(data_path)

    summary_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        name = str(scenario["name"])
        scenario_config = dict(config)
        scenario_config["save_dir"] = str(save_root_path / name)
        scenario_config["include_sites"] = scenario.get("include_sites")
        scenario_config["holdout_site"] = scenario.get("holdout_site")

        print(f"\n{'#' * 72}")
        print(f"DOMAIN SHIFT SCENARIO: {name}")
        print(f"{'#' * 72}")
        result = train(**scenario_config)
        summary_rows.append(_domain_shift_summary_row(name, result))
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = save_root_path / "domain_shift_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    if fieldnames:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nDomain-shift summary saved -> {csv_path}")

    pkl_path = save_root_path / "domain_shift_summary.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(summary_rows, handle)
    print(f"Domain-shift summary pickle saved -> {pkl_path}")

    print("\nDOMAIN SHIFT SUMMARY")
    print("Scenario                 ValSubBal  TestSubBal  TestSegBal  AUC")
    print("----------------------------------------------------------------")
    for row in summary_rows:
        print(
            f"{row['scenario']:<24} "
            f"{row['best_val_subject_balanced_acc']:>8.2f}% "
            f"{row['test_subject_balanced_acc']:>10.2f}% "
            f"{row['test_segment_balanced_acc']:>10.2f}% "
            f"{row['test_auc_roc']:>5.3f}"
        )
    return summary_rows


def train_per_site_independent() -> None:
    """
    Reproduces the per-site 32-channel validation reported in the R2R
    (Reviewer 2, Comment 4, Table 3): each dataset is trained and evaluated
    INDEPENDENTLY on its own 70:15:15 split (not a joint 3-site model).
    Site-adversarial training is off for this — there is no cross-site
    domain to be invariant to within a single-site run.
    """
    for site in ("ucsd", "unm", "iowa"):
        print(f"\n{'='*70}\n  Training on {site.upper()} only (32-channel, 70:15:15 split)\n{'='*70}")
        cfg = dict(DEFAULT_TRAIN_CONFIG)
        cfg["include_sites"] = site
        cfg["use_site_adversarial"] = False
        cfg["save_dir"] = f"./results_32_{site}"
        train(**cfg)


def train_joint_all_sites() -> None:
    """
    Trains ONE model jointly on all three datasets combined (UCSD + UNM +
    Iowa), 32-channel common montage, 70:15:15 subject-level train/val/test
    split drawn across all three sites together (not per-site). Site-
    adversarial training is enabled — with three sites mixed into one
    training set, this is what prevents the model from using site identity
    as a shortcut instead of the genuine PD/HC signal (see the site-
    adversarial diagnosis worked through for the 40-channel pipeline).
    """
    print(f"\n{'='*70}\n  Training jointly on UCSD + UNM + Iowa "
          f"(32-channel, 70:15:15 split)\n{'='*70}")
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    cfg["include_sites"] = None       # use all three sites together
    cfg["use_site_adversarial"] = True
    cfg["save_dir"] = "./results_32_joint"
    train(**cfg)


def main() -> None:
    train_joint_all_sites()


if __name__ == "__main__":
    main()
