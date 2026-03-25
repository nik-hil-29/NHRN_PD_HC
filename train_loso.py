"""
train_loso.py
=============
Leave-One-Subject-Out (LOSO) cross-validation training for NHRN.
  - LOSO over all 31 subjects (16 HC + 15 PD)
  - AdamW, lr=1e-3, weight_decay=1e-3, batch_size=32
  - Binary cross-entropy loss  (Equation 24)
  - Early stopping on validation loss
  - Saves per-fold results and final aggregated metrics
"""

import os
import time
import pickle
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, cohen_kappa_score,
                             matthews_corrcoef, precision_score, recall_score)

from model import build_model, build_optimizer, binary_cross_entropy_loss

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── dataset ────────────────────────────────────────────────────────────────────

class EEGSegmentDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()   # (N, 40, 1024)
        self.y = torch.from_numpy(y).float()   # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Balanced sampler to handle class imbalance within folds."""
    counts  = np.bincount(y.astype(int))
    weights = 1.0 / counts[y.astype(int)]
    return WeightedRandomSampler(torch.from_numpy(weights).double(),
                                 num_samples=len(weights), replacement=True)


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """
    y_true, y_pred: binary int arrays
    y_prob:         probability of class 1 (PD)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv         = tp / (tp + fp + 1e-9)
    npv         = tn / (tn + fn + 1e-9)
    lr_pos      = sensitivity / (1 - specificity + 1e-9)
    lr_neg      = (1 - sensitivity) / (specificity + 1e-9)
    dor         = lr_pos / (lr_neg + 1e-9)

    # Number needed to diagnose
    nnd = 1 / (sensitivity - (1 - specificity) + 1e-9)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'accuracy':    accuracy_score(y_true, y_pred) * 100,
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'auc_roc':     auc,
        'kappa':       cohen_kappa_score(y_true, y_pred),
        'mcc':         matthews_corrcoef(y_true, y_pred),
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'recall':      recall_score(y_true, y_pred, zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv':         ppv,
        'npv':         npv,
        'lr_pos':      lr_pos,
        'lr_neg':      lr_neg,
        'dor':         dor,
        'nnd':         nnd,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ── training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = binary_cross_entropy_loss(pred, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (loss, y_pred, y_prob)."""
    model.eval()
    all_probs = []
    all_true  = []
    total_loss = 0.0
    n_batches  = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        prob = model(X_b)
        loss = binary_cross_entropy_loss(prob, y_b)
        total_loss += loss.item()
        n_batches  += 1
        all_probs.append(prob.cpu().numpy())
        all_true.append(y_b.cpu().numpy())

    probs = np.concatenate(all_probs)
    trues = np.concatenate(all_true).astype(int)
    preds = (probs >= 0.5).astype(int)
    return total_loss / n_batches, preds, probs


# ── LOSO cross-validation ──────────────────────────────────────────────────────

def loso_cv(X: np.ndarray,
            y: np.ndarray,
            subjects: np.ndarray,
            sessions: np.ndarray,
            metadata: dict,
            device: str = 'cpu',
            epochs: int = 50,
            patience: int = 10,
            batch_size: int = 32,
            lr: float = 1e-3,
            weight_decay: float = 1e-3,
            dropout: float = 0.3,
            save_dir: str = './results') -> dict:
    """
    Leave-One-Subject-Out cross-validation.

    One fold = one subject held out for testing; remaining subjects used for
    training (with a 10% internal validation split for early stopping).
    """
    os.makedirs(save_dir, exist_ok=True)

    unique_subjects = sorted(np.unique(subjects))
    n_subjects      = len(unique_subjects)
    print(f"\nLOSO-CV: {n_subjects} subjects | device: {device}")
    print("=" * 65)

    fold_results = {}
    all_preds    = []
    all_trues    = []
    all_probs    = []
    all_sub_ids  = []

    total_start = time.time()

    for fold_idx, test_sub in enumerate(unique_subjects):
        print(f"\nFold {fold_idx+1:>2}/{n_subjects}  test_subject={test_sub}")

        # ── split segments ─────────────────────────────────────────────────
        test_mask  = (subjects == test_sub)
        train_mask = ~test_mask

        X_test, y_test = X[test_mask], y[test_mask]

        X_tr_all = X[train_mask]
        y_tr_all = y[train_mask]

        # 10% internal validation from training set (different subjects)
        tr_subs = subjects[train_mask]
        val_subs = sorted(np.unique(tr_subs))
        n_val   = max(1, int(0.1 * len(val_subs)))
        val_sub_list = val_subs[:n_val]
        val_mask_tr  = np.isin(tr_subs, val_sub_list)

        X_val, y_val = X_tr_all[val_mask_tr], y_tr_all[val_mask_tr]
        X_tr,  y_tr  = X_tr_all[~val_mask_tr], y_tr_all[~val_mask_tr]

        print(f"  train={len(X_tr):>4} | val={len(X_val):>4} | test={len(X_test):>4} segs")

        # ── dataloaders ────────────────────────────────────────────────────
        sampler    = make_sampler(y_tr)
        train_ds   = EEGSegmentDataset(X_tr,   y_tr)
        val_ds     = EEGSegmentDataset(X_val,  y_val)
        test_ds    = EEGSegmentDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=2,
                                  pin_memory=(device == 'cuda'))
        val_loader   = DataLoader(val_ds, batch_size=batch_size,
                                  shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds, batch_size=batch_size,
                                  shuffle=False, num_workers=2)

        # ── model + optimizer ──────────────────────────────────────────────
        model     = build_model(device=device, dropout=dropout)
        optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 10)

        # ── training with early stopping ───────────────────────────────────
        best_val_loss = float('inf')
        best_state    = None
        no_improve    = 0
        fold_start    = time.time()

        for epoch in range(1, epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, device)
            va_loss, _, _ = evaluate(model, val_loader, device)
            scheduler.step()

            if va_loss < best_val_loss - 1e-5:
                best_val_loss = va_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (no improve {patience} epochs)")
                break

            if epoch % 10 == 0 or epoch == 1:
                print(f"  ep {epoch:3d} | tr_loss={tr_loss:.4f} | "
                      f"val_loss={va_loss:.4f} | best={best_val_loss:.4f}")

        fold_time = time.time() - fold_start

        # ── evaluate on test subject ───────────────────────────────────────
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, preds, probs = evaluate(model, test_loader, device)
        true            = y_test.astype(int)

        metrics = compute_metrics(true, preds, probs)
        label   = 'PD' if y_test[0] == 1 else 'HC'

        # UPDRS info
        sub_meta = metadata.get(test_sub, {})
        updrs_on  = sub_meta.get('updrs_on',  None)
        updrs_off = sub_meta.get('updrs_off', None)
        ses_list  = [s['session'] for s in sub_meta.get('sessions', [])]
        ses       = ses_list[0] if ses_list else 'unknown'

        fold_results[test_sub] = {
            'subject': test_sub,
            'label':   label,
            'session': ses,
            'updrs_on': updrs_on, 'updrs_off': updrs_off,
            'n_test_segs': len(y_test),
            'metrics': metrics,
            'time_s': fold_time,
        }

        all_preds.extend(preds.tolist())
        all_trues.extend(true.tolist())
        all_probs.extend(probs.tolist())
        all_sub_ids.extend([test_sub] * len(true))

        print(f"  → acc={metrics['accuracy']:.2f}%  "
              f"F1={metrics['f1']:.4f}  "
              f"AUC={metrics['auc_roc']:.4f}  "
              f"[{fold_time:.1f}s]")

        # Save fold model
        torch.save(best_state,
                   os.path.join(save_dir, f'model_{test_sub}.pt'))

    # ── aggregate results ──────────────────────────────────────────────────
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    all_probs = np.array(all_probs)

    overall = compute_metrics(all_trues, all_preds, all_probs)
    total_time = time.time() - total_start

    # Per-subject accuracy summary
    per_sub_acc = {}
    for sub in unique_subjects:
        mask = np.array(all_sub_ids) == sub
        if mask.sum() > 0:
            per_sub_acc[sub] = accuracy_score(
                all_trues[mask], all_preds[mask]) * 100

    print("\n" + "=" * 65)
    print("LOSO-CV OVERALL RESULTS")
    print("=" * 65)
    print(f"  Accuracy    : {overall['accuracy']:.2f}%")
    print(f"  F1-Score    : {overall['f1']:.4f}")
    print(f"  AUC-ROC     : {overall['auc_roc']:.4f}")
    print(f"  Cohen κ     : {overall['kappa']:.4f}")
    print(f"  MCC         : {overall['mcc']:.4f}")
    print(f"  Sensitivity : {overall['sensitivity']:.4f}")
    print(f"  Specificity : {overall['specificity']:.4f}")
    print(f"  PPV         : {overall['ppv']:.4f}")
    print(f"  NPV         : {overall['npv']:.4f}")
    print(f"  LR+         : {overall['lr_pos']:.2f}")
    print(f"  LR-         : {overall['lr_neg']:.4f}")
    print(f"  DOR         : {overall['dor']:.1f}")
    print(f"  Total time  : {total_time/3600:.2f} h")
    print("=" * 65)

    # Per-subject table (like Table 5)
    print("\nPer-subject results:")
    print(f"  {'Subject':<12} {'Label':>5} {'N_segs':>7} "
          f"{'Acc(%)':>8} {'F1':>7} {'AUC':>7} {'UPDRS_ON':>9} {'UPDRS_OFF':>10}")
    print("  " + "-" * 73)
    for sub in unique_subjects:
        r = fold_results[sub]
        m = r['metrics']
        print(f"  {sub:<12} {r['label']:>5} {r['n_test_segs']:>7} "
              f"{m['accuracy']:>8.2f} {m['f1']:>7.4f} {m['auc_roc']:>7.4f} "
              f"{str(r['updrs_on']):>9} {str(r['updrs_off']):>10}")

    summary = {
        'fold_results':  fold_results,
        'overall':       overall,
        'all_preds':     all_preds,
        'all_trues':     all_trues,
        'all_probs':     all_probs,
        'all_sub_ids':   all_sub_ids,
        'per_sub_acc':   per_sub_acc,
        'total_time_s':  total_time,
    }

    with open(os.path.join(save_dir, 'loso_results.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nResults saved to {save_dir}/loso_results.pkl")

    return summary


# ── multi-fold cross validation (5-fold, 10-fold) ─────────────────────────────

def kfold_cv(X, y, subjects, device='cpu', k=5, **train_kwargs):
    """
    k-fold CV over subjects (not segments) for Table 8 comparison.
    Returns aggregate metrics.
    """
    from sklearn.model_selection import KFold
    unique_subs = sorted(np.unique(subjects))
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)

    all_preds, all_trues, all_probs = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(unique_subs)):
        print(f"\n  k-fold {fold+1}/{k}")
        tr_subs = [unique_subs[i] for i in tr_idx]
        te_subs = [unique_subs[i] for i in te_idx]

        tr_mask = np.isin(subjects, tr_subs)
        te_mask = np.isin(subjects, te_subs)

        X_tr, y_tr   = X[tr_mask], y[tr_mask]
        X_te, y_te   = X[te_mask], y[te_mask]

        sampler   = make_sampler(y_tr)
        train_ds  = EEGSegmentDataset(X_tr, y_tr)
        test_ds   = EEGSegmentDataset(X_te, y_te)

        bs = train_kwargs.get('batch_size', 32)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

        model     = build_model(device=device)
        optimizer = build_optimizer(model)
        epochs    = train_kwargs.get('epochs', 50)

        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, device)

        _, preds, probs = evaluate(model, test_loader, device)
        all_preds.extend(preds.tolist())
        all_trues.extend(y_te.astype(int).tolist())
        all_probs.extend(probs.tolist())

    return compute_metrics(np.array(all_trues),
                           np.array(all_preds),
                           np.array(all_probs))


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./preprocessed')
    parser.add_argument('--save_dir',   default='./results')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--dropout',    type=float, default=0.3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load preprocessed data
    from preprocessing import load_preprocessed
    data = load_preprocessed(args.data_dir)

    X        = data['segments']          # (N, 40, 1024)
    y        = data['labels']            # (N,)
    subjects = data['subjects']          # (N,) subject IDs
    sessions = data['sessions']          # (N,)
    metadata = data['metadata']

    print(f"Loaded: X={X.shape}  y={y.shape}  subjects={len(np.unique(subjects))}")

    # Run LOSO
    results = loso_cv(
        X=X, y=y, subjects=subjects, sessions=sessions,
        metadata=metadata, device=device,
        epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, dropout=args.dropout,
        save_dir=args.save_dir
    )
