"""
core/train.py — IMPROVED training pipeline v3
Changes from v2:
  - Uses MultiScaleTBModel throughout (not TBModel)
  - Removed freeze_backbone() — manual per-stage freeze for multi-scale
  - pretrained=False when loading saved weights for post-training steps
  - num_workers=0 (Windows safe)
  - focal_alpha = 0.54 (tuned to actual class ratio: 8513/15776)
  - label_smooth = 0.1 (prevents overconfidence)
  - patience = 15 (lets highres stage run)
  - xray_aug = True (gamma, elastic, random CLAHE)
  - weight_decay = 2e-4 (stabilises val loss spikes)
  - Runs threshold finder + temperature scaling after training
"""

from __future__ import annotations
import os, sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import numpy as np

from core.model         import (FocalLoss, find_optimal_threshold,
                                TemperatureScaler, MultiScaleTBModel)
from core.preprocessing import TBDataset, mixup, cutmix, mix_loss


# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

# Tuned to your exact dataset: TB=8513, Normal=7263, total=15776
# alpha = TB_count / total = 8513/15776 = 0.54
ALPHA_TUNED = round(8513 / 15776, 4)   # 0.5396

STAGES = [
    # name        epochs  size   lr      batch  freeze  aug
    ("warmup",    8,      224,   3e-4,   64,    True,   False),
    ("finetune",  25,     224,   1e-4,   64,    False,  True),
    ("highres",   15,     384,   5e-5,   32,    False,  True),
]

DEFAULTS = dict(
    data_root    = "data/",
    ckpt_dir     = "checkpoints/",
    num_classes  = 2,
    focal_gamma  = 2.0,
    focal_alpha  = ALPHA_TUNED,
    label_smooth = 0.1,
    weight_decay = 2e-4,
    aug_prob     = 0.5,
    patience     = 15,
    num_workers  = 0,           # Windows safe
    amp          = True,
    xray_aug     = True,
)


# ══════════════════════════════════════════════════════════════
#  BALANCED SAMPLER
# ══════════════════════════════════════════════════════════════

def balanced_sampler(dataset: TBDataset) -> WeightedRandomSampler:
    labels  = [s[1] for s in dataset.samples]
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights     = torch.DoubleTensor(weights),
        num_samples = len(dataset),
        replacement = True,
    )


# ══════════════════════════════════════════════════════════════
#  FREEZE HELPERS — per-stage for MultiScaleTBModel
# ══════════════════════════════════════════════════════════════

def freeze_for_warmup(model: MultiScaleTBModel):
    """
    Stage 1 — Warmup:
    Freeze early + mid backbone.
    Keep late_backbone, attention, fusion, proj, head trainable.
    """
    for param in model.early_backbone.parameters():
        param.requires_grad = False
    for param in model.mid_backbone.parameters():
        param.requires_grad = False
    for param in model.late_backbone.parameters():
        param.requires_grad = True
    for param in model.early_attn.parameters():
        param.requires_grad = False
    for param in model.mid_attn.parameters():
        param.requires_grad = False
    for param in model.late_attn.parameters():
        param.requires_grad = True
    for param in model.fusion.parameters():
        param.requires_grad = True
    for param in model.proj.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True


def unfreeze_all(model: MultiScaleTBModel):
    """Stage 2 + 3 — all parameters trainable."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model: MultiScaleTBModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch

            if labels.dim() == 2:
                hard_labels = labels.argmax(1).to(device)
            else:
                hard_labels = labels.to(device)
            imgs = imgs.to(device)

            logits, _ = model(imgs, None)
            loss      = criterion(logits, hard_labels)
            total_loss += loss.item()
            preds      = logits.argmax(1)
            correct   += (preds == hard_labels).sum().item()
            total     += hard_labels.size(0)
            probs      = torch.softmax(logits, 1)[:, 1]
            all_probs  += probs.cpu().tolist()
            all_labels += hard_labels.cpu().tolist()

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return {
        "loss": total_loss / len(loader),
        "acc":  correct / total,
        "auc":  auc,
    }


# ══════════════════════════════════════════════════════════════
#  TRAINING STAGE
# ══════════════════════════════════════════════════════════════

def run_stage(model, tr_loader, val_loader, cfg, name, epochs, lr,
              use_aug, device, best_auc, patience_ctr):

    criterion = FocalLoss(cfg["focal_gamma"], cfg["focal_alpha"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs, eta_min=1e-6)

    use_amp = cfg["amp"] and device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None
    history = []

    trainable = count_trainable(model)
    print(f"  Trainable params: {trainable/1e6:.2f}M", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        t0      = time.time()
        n_batch = len(tr_loader)

        for batch_idx, batch in enumerate(tr_loader, 1):
            imgs, labels = batch

            if labels.dim() == 2:
                soft_labels = labels.to(device)
                hard_labels = labels.argmax(1).to(device)
            else:
                hard_labels = labels.to(device)
                soft_labels = None

            imgs = imgs.to(device)

            use_mix = (use_aug
                       and torch.rand(1).item() < cfg["aug_prob"]
                       and soft_labels is None)
            if use_mix:
                fn = mixup if torch.rand(1).item() < 0.5 else cutmix
                imgs, ya, yb, lam = fn(imgs, hard_labels)

            optimizer.zero_grad()

            def forward_loss():
                logits, _ = model(imgs, None)
                if use_mix:
                    return mix_loss(criterion, logits, ya, yb, lam)
                if soft_labels is not None:
                    log_probs = torch.log_softmax(logits, dim=1)
                    return -(soft_labels * log_probs).sum(dim=1).mean()
                return criterion(logits, hard_labels)

            if scaler:
                with autocast("cuda"):
                    loss = forward_loss()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ep_loss += loss.item()

            if batch_idx % 10 == 0 or batch_idx == n_batch:
                pct = batch_idx / n_batch * 100
                bar = ("█" * int(pct // 5)).ljust(20)
                print(f"\r  [{name}] Epoch {ep:02}/{epochs}  "
                      f"|{bar}| {batch_idx}/{n_batch}  "
                      f"loss={ep_loss/batch_idx:.4f}  ({pct:.0f}%)",
                      end="", flush=True)

        scheduler.step()
        val     = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"\r[{name}] Epoch {ep:02}/{epochs}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}  "
              f"train_loss={ep_loss/n_batch:.4f}  "
              f"val_loss={val['loss']:.4f}  "
              f"acc={val['acc']:.4f}  "
              f"auc={val['auc']:.4f}  "
              f"time={elapsed:.0f}s",
              flush=True)

        history.append({
            "stage":   name,
            "epoch":   ep,
            "val_acc": val["acc"],
            "val_auc": val["auc"],
        })

        if val["auc"] > best_auc:
            best_auc     = val["auc"]
            patience_ctr = 0
            path = Path(cfg["ckpt_dir"]) / "best_model.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"  ✓ New best AUC {best_auc:.4f} → {path}", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"  Early stop (patience={cfg['patience']}).", flush=True)
                break

    return history, best_auc, patience_ctr


# ══════════════════════════════════════════════════════════════
#  MAIN TRAIN FUNCTION
# ══════════════════════════════════════════════════════════════

def train(config: dict = {}):
    cfg    = {**DEFAULTS, **config}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*56}", flush=True)
    print(f"TB Detection Training v3  —  {device}", flush=True)
    print(f"Model       : MultiScaleTBModel (3-scale EfficientNetV2-S)", flush=True)
    print(f"focal_alpha : {cfg['focal_alpha']:.4f}  (tuned to class ratio)", flush=True)
    print(f"label_smooth: {cfg['label_smooth']}", flush=True)
    print(f"patience    : {cfg['patience']}", flush=True)
    print(f"{'='*56}\n", flush=True)

    # ── Build model (pretrained=True loads ImageNet weights) ──
    model = MultiScaleTBModel(
        num_classes  = cfg["num_classes"],
        pretrained   = True,
        dropout      = 0.4,
        fusion_dim   = 256,
        use_metadata = False,
    )
    model = model.to(device)

    best_auc     = 0.0
    patience_ctr = 0
    all_history  = []

    # ── Run each stage ────────────────────────────────────────
    for name, epochs, size, lr, batch, freeze, use_aug in STAGES:
        print(f"\n── Stage: {name.upper()}  "
              f"size={size}  lr={lr}  epochs={epochs}  batch={batch} ──",
              flush=True)

        if freeze:
            freeze_for_warmup(model)
            print("  Freeze: early_backbone + mid_backbone frozen", flush=True)
        else:
            unfreeze_all(model)
            print("  Freeze: all parameters trainable", flush=True)

        tr_ds = TBDataset(
            cfg["data_root"], "train", size=size,
            label_smooth = cfg["label_smooth"] if use_aug else 0.0,
            xray_aug     = cfg["xray_aug"] and use_aug,
        )
        va_ds = TBDataset(cfg["data_root"], "val", size=size)

        tr_loader = DataLoader(
            tr_ds,
            batch_size  = batch,
            sampler     = balanced_sampler(tr_ds),
            num_workers = cfg["num_workers"],
            pin_memory  = device.type == "cuda",
        )
        va_loader = DataLoader(
            va_ds,
            batch_size  = batch,
            shuffle     = False,
            num_workers = cfg["num_workers"],
            pin_memory  = device.type == "cuda",
        )

        hist, best_auc, patience_ctr = run_stage(
            model, tr_loader, va_loader, cfg,
            name, epochs, lr, use_aug, device,
            best_auc, patience_ctr,
        )
        all_history  += hist
        patience_ctr  = 0   # reset between stages

    # ── Save training history ─────────────────────────────────
    hist_path = Path(cfg["ckpt_dir"]) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(all_history, f, indent=2)

    print(f"\n{'='*56}", flush=True)
    print(f"Training complete. Best AUC: {best_auc:.4f}", flush=True)
    print(f"{'='*56}\n", flush=True)

    # ══════════════════════════════════════════════════════════
    #  POST-TRAINING: LOAD BEST WEIGHTS
    #  pretrained=False — loading OUR weights, not ImageNet
    # ══════════════════════════════════════════════════════════
    print("Loading best checkpoint for post-training steps...", flush=True)

    best_model = MultiScaleTBModel(
        num_classes  = 2,
        pretrained   = False,       # ← must be False here
        dropout      = 0.4,
        fusion_dim   = 256,
        use_metadata = False,
    )
    best_model.load_state_dict(
        torch.load(
            Path(cfg["ckpt_dir"]) / "best_model.pth",
            map_location=device,
        )
    )
    best_model = best_model.to(device)
    best_model.eval()
    print("Best checkpoint loaded successfully.", flush=True)

    # ══════════════════════════════════════════════════════════
    #  POST-TRAINING: OPTIMAL THRESHOLD (Youden's J)
    # ══════════════════════════════════════════════════════════
    print("\nRunning threshold optimisation on validation set...", flush=True)

    va_ds_final     = TBDataset(cfg["data_root"], "val", size=224)
    va_loader_final = DataLoader(
        va_ds_final,
        batch_size  = 32,
        shuffle     = False,
        num_workers = 0,            # Windows safe
    )

    threshold_results = find_optimal_threshold(
        best_model, va_loader_final,
        device = str(device),
        metric = "youden",
    )

    threshold_path = Path(cfg["ckpt_dir"]) / "optimal_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "optimal_threshold": threshold_results["optimal_threshold"],
            "sensitivity":       threshold_results["sensitivity"],
            "specificity":       threshold_results["specificity"],
            "f1":                threshold_results["f1"],
            "auc":               threshold_results["auc"],
            "accuracy":          threshold_results["accuracy"],
            "threshold_method":  "youden_j",
            "model":             "MultiScaleTBModel",
        }, f, indent=2)

    print(f"Optimal threshold : {threshold_results['optimal_threshold']:.4f}", flush=True)
    print(f"  Sensitivity     : {threshold_results['sensitivity']:.4f}", flush=True)
    print(f"  Specificity     : {threshold_results['specificity']:.4f}", flush=True)
    print(f"  F1              : {threshold_results['f1']:.4f}", flush=True)
    print(f"  AUC             : {threshold_results['auc']:.4f}", flush=True)
    print(f"Saved → {threshold_path}", flush=True)

    # ══════════════════════════════════════════════════════════
    #  POST-TRAINING: TEMPERATURE SCALING
    # ══════════════════════════════════════════════════════════
    print("\nFitting temperature scaling...", flush=True)

    temp_scaler = TemperatureScaler(best_model)
    temp_scaler.fit(va_loader_final, device=str(device))

    t_path = Path(cfg["ckpt_dir"]) / "temperature_scaler.pth"
    torch.save(temp_scaler.state_dict(), t_path)
    print(f"Temperature T = {temp_scaler.temperature.item():.4f}", flush=True)
    print(f"Saved → {t_path}", flush=True)

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*56}", flush=True)
    print(f"  ALL DONE", flush=True)
    print(f"  Best AUC          : {best_auc:.4f}", flush=True)
    print(f"  Optimal threshold : {threshold_results['optimal_threshold']:.4f}", flush=True)
    print(f"  Temperature T     : {temp_scaler.temperature.item():.4f}", flush=True)
    print(f"  Checkpoint        : checkpoints/best_model.pth", flush=True)
    print(f"  Threshold file    : checkpoints/optimal_threshold.json", flush=True)
    print(f"  Temperature file  : checkpoints/temperature_scaler.pth", flush=True)
    print(f"{'='*56}", flush=True)
    print(f"\nNext: streamlit run tb_system/app/app.py", flush=True)

    return best_auc, threshold_results["optimal_threshold"]


if __name__ == "__main__":
    train()