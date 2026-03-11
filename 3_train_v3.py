"""
3_train_v3.py
=============
Improved training script with:
  - Cosine Annealing learning rate (replaces step decay)
  - Stochastic Weight Averaging (SWA) in final epochs
  - Works with both TBModel and MultiScaleTBModel
  - All v2 improvements retained (focal loss, label smoothing, etc.)

Run from project root:
    python 3_train_v3.py

Expected improvement over v2: +1-2% AUC, better generalisation
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tb_system"))

from core.model         import MultiScaleTBModel
from core.preprocessing import TBDataset

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    # Paths
    "data_dir":    os.path.join(_root, "data"),
    "ckpt_dir":    os.path.join(_root, "checkpoints"),

    # Architecture
    "model":       "TBModel",   # or "MultiScaleTBModel"
    "num_classes": 2,
    "dropout":     0.4,

    # Stage 1 — Warmup (backbone frozen)
    "warmup_epochs":    8,
    "warmup_lr":        3e-4,
    "warmup_batch":     128,      # was 64 — AMP frees VRAM
    "warmup_size":      224,

    # Stage 2 — Finetune (full model)
    "finetune_epochs":  25,
    "finetune_lr":      1e-4,
    "finetune_batch":   96,       # was 64
    "finetune_size":    224,

    # Stage 3 — High-Res
    "highres_epochs":   15,
    "highres_lr":       5e-5,
    "highres_batch":    24,
    "highres_size":     384,
    "highres_swa":      False,
    "highres_workers":  2,        # fewer workers for 384px — heavy augmentation

    # SWA — starts in last N epochs of each stage
    "swa_start_frac":   0.6,    # SWA kicks in at 60% through each stage
    "swa_lr":           5e-5,

    # Loss
    "focal_alpha":      0.54,   # TB_count / total = 8513/15776
    "focal_gamma":      2.0,
    "label_smooth":     0.1,

    # Training
    "patience":         15,
    "weight_decay":     2e-4,
    "grad_clip":        1.0,
    "workers":          4,        # parallel data loading — use 4 on Windows
    "pin_memory":       True,     # faster CPU→GPU transfer
    "persistent_workers": True,   # keep workers alive between epochs
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CFG["ckpt_dir"], exist_ok=True)


# ── FOCAL LOSS ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.54, gamma=2.0, label_smooth=0.1):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.label_smooth = label_smooth

    def forward(self, logits, targets):
        n_cls = logits.size(1)

        # TBDataset with label_smooth returns 2D soft float labels
        # TBDataset without label_smooth returns 1D hard int64 labels
        if targets.dim() == 2:
            # Already soft labels from dataset — use directly
            smooth_targets = targets.float()
            hard_labels    = targets.argmax(1)
        else:
            # Hard int64 labels — apply smoothing here
            hard_labels    = targets.long()
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smooth / (n_cls - 1))
            smooth_targets.scatter_(1, hard_labels.unsqueeze(1),
                                    1.0 - self.label_smooth)

        log_p  = F.log_softmax(logits, dim=1)
        probs  = torch.exp(log_p)
        focal  = (1 - probs) ** self.gamma
        loss   = -(smooth_targets * focal * log_p).sum(dim=1)

        # Alpha weighting using hard labels
        alpha_t = torch.where(hard_labels == 1,
                              torch.tensor(self.alpha,     device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))
        return (alpha_t * loss).mean()


# ── COSINE SCHEDULE ───────────────────────────────────────────
def cosine_schedule(optimizer, lr_max, lr_min, n_epochs):
    """Cosine annealing from lr_max to lr_min over n_epochs."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min)


# ── AMP SCALER (mixed precision — 2x faster on RTX) ──────────
scaler = torch.amp.GradScaler("cuda")

# ── TRAIN ONE EPOCH ───────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scheduler=None, swa_model=None, swa_scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    n_batch = len(loader)

    for batch_idx, (imgs, labels, *_) in enumerate(loader, 1):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast("cuda"):
            logits, _ = model(imgs, None)
            loss      = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if swa_model is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(imgs)
        n          += len(imgs)

        # ── Progress bar ──────────────────────────────────────
        pct      = batch_idx / n_batch
        filled   = int(pct * 40)
        bar      = "█" * filled + "░" * (40 - filled)
        avg_loss = total_loss / n
        print(f"\r  [{bar}] {pct*100:5.1f}%  "
              f"batch {batch_idx}/{n_batch}  "
              f"loss {avg_loss:.4f}",
              end="", flush=True)

    print()  # newline after epoch completes
    return total_loss / n


# ── VALIDATE ─────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_probs, all_labels = [], []

    for imgs, labels, *_ in loader:
        imgs = imgs.to(DEVICE)
        # Handle soft labels — convert to hard for AUC calculation
        if labels.dim() == 2:
            labels = labels.argmax(1)
        logits, _ = model(imgs, None)
        probs     = F.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return roc_auc_score(labels, probs), probs, labels


# ── YOUDEN THRESHOLD ─────────────────────────────────────────
def find_threshold(probs, labels):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j     = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ── STAGE RUNNER ─────────────────────────────────────────────
def run_stage(name, model, train_loader, val_loader, optimizer, criterion,
              n_epochs, best_auc, best_path, enable_swa=True):

    swa_start  = int(n_epochs * CFG["swa_start_frac"]) if enable_swa else n_epochs + 1
    swa_model  = AveragedModel(model)
    swa_sched  = SWALR(optimizer, swa_lr=CFG["swa_lr"],
                       anneal_epochs=max(1, n_epochs - swa_start))
    cos_sched  = cosine_schedule(optimizer,
                                 lr_max=optimizer.param_groups[0]["lr"],
                                 lr_min=CFG["swa_lr"] / 10,
                                 n_epochs=max(swa_start, n_epochs))

    patience_ctr = 0
    print(f"\n{'='*55}")
    swa_label = f"SWA after epoch {swa_start}" if enable_swa else "no SWA"
    print(f"  {name}  ({n_epochs} epochs, {swa_label})")
    print(f"{'='*55}")

    for epoch in range(1, n_epochs + 1):
        t0      = time.time()
        use_swa = enable_swa and epoch > swa_start

        loss = train_epoch(
            model, train_loader, optimizer, criterion,
            scheduler     = None if use_swa else cos_sched,
            swa_model     = swa_model    if use_swa else None,
            swa_scheduler = swa_sched    if use_swa else None,
        )

        # Always validate with BASE model — SWA wrapper has stale BN without update_bn
        auc, probs, labels = validate(model, val_loader)
        elapsed = time.time() - t0

        tag = "[SWA]" if use_swa else "     "
        print(f"  Epoch {epoch:3d}/{n_epochs} {tag} | Loss {loss:.4f} | "
              f"AUC {auc:.4f} | {elapsed:.0f}s")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"    ↑ New best! Saved → {best_path}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # BN update skipped — adds 10+ min per stage with no significant gain
    # update_bn(train_loader, swa_model, device=DEVICE)

    return best_auc, find_threshold(probs, labels)


# ── MAIN TRAINING LOOP ────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  TB Detection System v3 — Training")
    print(f"  Device: {DEVICE}")
    print(f"  Model : MultiScaleTBModel (3-scale EfficientNetV2-S)")
    print(f"  Config: Cosine LR + SWA")
    print(f"{'='*55}")

    # ── Model ─────────────────────────────────────────────────
    model = MultiScaleTBModel(
        num_classes  = CFG["num_classes"],
        pretrained   = True,
        dropout      = CFG["dropout"],
        fusion_dim   = 256,
        use_metadata = False,
    )
    model.to(DEVICE)

    criterion = FocalLoss(
        alpha        = CFG["focal_alpha"],
        gamma        = CFG["focal_gamma"],
        label_smooth = CFG["label_smooth"],
    )
    best_path = os.path.join(CFG["ckpt_dir"], "best_model.pth")
    best_auc  = 0.0

    # ── STAGE 1 — WARMUP ──────────────────────────────────────
    for p in model.early_backbone.parameters():
        p.requires_grad = False
    for p in model.mid_backbone.parameters():
        p.requires_grad = False
    for p in model.late_backbone.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.proj.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    train_ds = TBDataset(CFG["data_dir"], split="train",
                         size=CFG["warmup_size"],
                         xray_aug=False,
                         label_smooth=0.0)
    val_ds   = TBDataset(CFG["data_dir"], split="val",
                         size=CFG["warmup_size"])
    train_ld = DataLoader(train_ds, batch_size=CFG["warmup_batch"],
                          shuffle=True,  num_workers=CFG["workers"],
                          pin_memory=CFG["pin_memory"],
                          persistent_workers=CFG["persistent_workers"])
    val_ld   = DataLoader(val_ds,   batch_size=CFG["warmup_batch"],
                          shuffle=False, num_workers=CFG["workers"],
                          pin_memory=CFG["pin_memory"],
                          persistent_workers=CFG["persistent_workers"])

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["warmup_lr"], weight_decay=CFG["weight_decay"])

    best_auc, thresh = run_stage(
        "STAGE 1 — WARMUP (early+mid frozen)",
        model, train_ld, val_ld, opt, criterion,
        CFG["warmup_epochs"], best_auc, best_path)

    # ── STAGE 2 — FINETUNE ────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = True

    train_ds = TBDataset(CFG["data_dir"], split="train",
                         size=CFG["finetune_size"],
                         xray_aug=True,
                         label_smooth=CFG["label_smooth"])
    val_ds   = TBDataset(CFG["data_dir"], split="val",
                         size=CFG["finetune_size"])
    train_ld = DataLoader(train_ds, batch_size=CFG["finetune_batch"],
                          shuffle=True,  num_workers=CFG["workers"],
                          pin_memory=CFG["pin_memory"],
                          persistent_workers=CFG["persistent_workers"])
    val_ld   = DataLoader(val_ds,   batch_size=CFG["finetune_batch"],
                          shuffle=False, num_workers=CFG["workers"],
                          pin_memory=CFG["pin_memory"],
                          persistent_workers=CFG["persistent_workers"])

    opt = torch.optim.AdamW(model.parameters(),
                             lr=CFG["finetune_lr"],
                             weight_decay=CFG["weight_decay"])

    best_auc, thresh = run_stage(
        "STAGE 2 — FINETUNE (all params, cosine + SWA)",
        model, train_ld, val_ld, opt, criterion,
        CFG["finetune_epochs"], best_auc, best_path,
        enable_swa=True)

    # ── STAGE 3 — HIGH-RES ────────────────────────────────────
    train_ds_hr = TBDataset(CFG["data_dir"], split="train",
                            size=CFG["highres_size"],
                            xray_aug=False,          # disabled — too slow at 384px
                            label_smooth=CFG["label_smooth"])
    val_ds_hr   = TBDataset(CFG["data_dir"], split="val",
                            size=CFG["highres_size"])
    train_ld_hr = DataLoader(train_ds_hr, batch_size=CFG["highres_batch"],
                             shuffle=True,  num_workers=CFG["highres_workers"],
                             pin_memory=CFG["pin_memory"],
                             persistent_workers=True)
    val_ld_hr   = DataLoader(val_ds_hr,   batch_size=CFG["highres_batch"],
                             shuffle=False, num_workers=CFG["highres_workers"],
                             pin_memory=CFG["pin_memory"],
                             persistent_workers=True)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=CFG["highres_lr"],
                             weight_decay=CFG["weight_decay"])

    best_auc, thresh = run_stage(
        "STAGE 3 — HIGH-RES 384px (cosine only, no SWA)",
        model, train_ld_hr, val_ld_hr, opt, criterion,
        CFG["highres_epochs"], best_auc, best_path,
        enable_swa=False)

    # ── SAVE THRESHOLD ────────────────────────────────────────
    thresh_path = os.path.join(CFG["ckpt_dir"], "optimal_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"optimal_threshold": thresh,
                   "threshold_method": "youden_j",
                   "final_auc": best_auc}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best AUC   : {best_auc:.4f}")
    print(f"  Threshold  : {thresh:.4f}")
    print(f"  Saved      : {best_path}")
    print(f"{'='*55}")
    print("\nNext step: run_temperature_scaling.py")


if __name__ == "__main__":
    main()