"""
core/model.py  —  Final TB Detection Model
============================================
Architecture : EfficientNetV2-S + SE-CBAM + Multi-Scale Fusion
Training     : Focal Loss + Supervised Contrastive Loss
GradCAM++    : Class-discriminative (TB vs Normal separate maps)
               Lung-masked, multi-scale weighted combination
Calibration  : Temperature Scaling (post-training)
TTA          : 8-fold test-time augmentation
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import Optional, Tuple, Dict


# ══════════════════════════════════════════════════════════════
#  ATTENTION BLOCKS
# ══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel-wise recalibration."""
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        mid = max(ch // r, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention: channel + spatial attention."""
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        mid = max(ch // r, 8)
        self.ch_avg  = nn.Sequential(nn.Linear(ch, mid, bias=False),
                                      nn.ReLU(), nn.Linear(mid, ch, bias=False))
        self.ch_max  = nn.Sequential(nn.Linear(ch, mid, bias=False),
                                      nn.ReLU(), nn.Linear(mid, ch, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.flatten(2).mean(-1)
        max_pool = x.flatten(2).max(-1).values
        ch_att   = torch.sigmoid(self.ch_avg(avg_pool) + self.ch_max(max_pool))
        x = x * ch_att.unsqueeze(-1).unsqueeze(-1)
        sp = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * torch.sigmoid(self.spatial(sp))


class SECBAMBlock(nn.Module):
    """SE followed by CBAM — suppresses non-lung regions."""
    def __init__(self, ch: int):
        super().__init__()
        self.se   = SEBlock(ch)
        self.cbam = CBAMBlock(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.se(x))


# ══════════════════════════════════════════════════════════════
#  SCALE FUSION
# ══════════════════════════════════════════════════════════════

class ScaleFusion(nn.Module):
    """
    Dynamic per-sample scale weighting.
    Learns to emphasise fine (nodule) or coarse (cavity) features
    depending on the image content.
    """
    def __init__(self, in_dims: list, out_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(d, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
            )
            for d in in_dims
        ])
        self.scale_attn = nn.Sequential(
            nn.Linear(out_dim * len(in_dims), 128),
            nn.GELU(),
            nn.Linear(128, len(in_dims)),
            nn.Softmax(dim=-1),
        )

    def forward(self, feature_maps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = [proj(fm) for proj, fm in zip(self.projs, feature_maps)]
        concat    = torch.cat(projected, dim=-1)
        weights   = self.scale_attn(concat)
        stacked   = torch.stack(projected, dim=1)
        fused     = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


# ══════════════════════════════════════════════════════════════
#  PROJECTION HEAD  — for contrastive loss only
# ══════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    Maps 512-d features to 128-d L2-normalised embedding space.
    Used ONLY during training for supervised contrastive loss.
    Forces TB and Normal clusters to be well-separated.
    """
    def __init__(self, in_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ══════════════════════════════════════════════════════════════
#  MAIN MODEL
# ══════════════════════════════════════════════════════════════

class MultiScaleTBModel(nn.Module):
    """
    EfficientNetV2-S + SE-CBAM + 3-scale fusion + contrastive projection head.

    Scale taps:
      early  blocks 0-2  : 48ch,   28x28  — small nodules, early infiltrates
      mid    blocks 3-5  : 160ch,  14x14  — consolidation, hilar involvement
      late   blocks 6-8  : 1280ch,  7x7  — cavities, global disease pattern
    """

    def __init__(
        self,
        num_classes:   int   = 2,
        pretrained:    bool  = True,
        dropout:       float = 0.4,
        fusion_dim:    int   = 256,
        use_metadata:  bool  = False,
        meta_dim:      int   = 16,
        use_proj_head: bool  = True,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        eff     = efficientnet_v2_s(weights=weights)
        blocks  = list(eff.features.children())

        self.early_backbone = nn.Sequential(*blocks[:3])
        self.mid_backbone   = nn.Sequential(*blocks[3:6])
        self.late_backbone  = nn.Sequential(*blocks[6:])

        self.early_attn = SECBAMBlock(48)
        self.mid_attn   = SECBAMBlock(160)
        self.late_attn  = SECBAMBlock(1280)

        self.fusion = ScaleFusion(in_dims=[48, 160, 1280], out_dim=fusion_dim)

        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout),
        )

        if use_metadata:
            self.meta_branch = nn.Sequential(
                nn.Linear(meta_dim, 64), nn.LayerNorm(64),
                nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
            )
            head_in = 512 + 128
        else:
            self.meta_branch = None
            head_in = 512

        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.proj_head = ProjectionHead(512, 128) if use_proj_head else None

    def forward(
        self,
        image:    torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        early = self.early_attn(self.early_backbone(image))
        mid   = self.mid_attn(self.mid_backbone(early))
        late  = self.late_attn(self.late_backbone(mid))

        fused, _  = self.fusion([early, mid, late])
        features  = self.proj(fused)

        if self.use_metadata and metadata is not None and self.meta_branch is not None:
            combined = torch.cat([features, self.meta_branch(metadata)], dim=1)
        else:
            combined = features

        return self.head(combined), features


# ══════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal loss with optional label smoothing."""
    def __init__(self, alpha: float = 0.54, gamma: float = 2.0,
                 label_smooth: float = 0.0):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        if targets.dim() == 2:
            smooth_t    = targets.float()
            hard_labels = targets.argmax(1)
        else:
            hard_labels = targets.long()
            smooth_t    = torch.zeros_like(logits)
            smooth_t.fill_(self.smooth / max(n_cls - 1, 1))
            smooth_t.scatter_(1, hard_labels.unsqueeze(1), 1.0 - self.smooth)

        log_p  = F.log_softmax(logits, dim=1)
        probs  = torch.exp(log_p)
        focal  = (1 - probs) ** self.gamma
        loss   = -(smooth_t * focal * log_p).sum(dim=1)
        alpha_t = torch.where(
            hard_labels == 1,
            torch.tensor(self.alpha,     device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device),
        )
        return (alpha_t * loss).mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls TB embeddings together, Normal embeddings together,
    pushes TB and Normal clusters apart in 128-d projection space.

    This is WHY GradCAM++ differentiates TB from Normal:
    the model is explicitly trained to make TB features look
    different from Normal features in embedding space, which
    forces spatially discriminative attention.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B      = features.size(0)
        if labels.dim() == 2:
            labels = labels.argmax(1)

        sim      = torch.matmul(features, features.T) / self.temperature
        mask_self = ~torch.eye(B, dtype=torch.bool, device=device)
        labels_e  = labels.unsqueeze(0).expand(B, -1)
        labels_c  = labels.unsqueeze(1).expand(-1, B)
        mask_pos  = (labels_e == labels_c) & mask_self

        exp_sim  = torch.exp(sim) * mask_self.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos    = mask_pos.float().sum(dim=1).clamp(min=1)
        loss     = -(log_prob * mask_pos.float()).sum(dim=1) / n_pos
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Focal + Supervised Contrastive combined.
    focal_w=1.0, contrastive_w=0.5 is a good starting point.
    """
    def __init__(
        self,
        focal_alpha:      float = 0.54,
        focal_gamma:      float = 2.0,
        label_smooth:     float = 0.0,
        contrastive_temp: float = 0.07,
        focal_w:          float = 1.0,
        contrastive_w:    float = 0.5,
    ):
        super().__init__()
        self.focal         = FocalLoss(focal_alpha, focal_gamma, label_smooth)
        self.contrastive   = SupervisedContrastiveLoss(contrastive_temp)
        self.focal_w       = focal_w
        self.contrastive_w = contrastive_w

    def forward(
        self,
        logits:     torch.Tensor,
        targets:    torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_loss = self.focal(logits, targets)
        if embeddings is not None and self.contrastive_w > 0:
            c_loss = self.contrastive(embeddings, targets)
            total  = self.focal_w * f_loss + self.contrastive_w * c_loss
        else:
            c_loss = torch.tensor(0.0, device=logits.device)
            total  = f_loss
        return total, f_loss, c_loss


# ══════════════════════════════════════════════════════════════
#  LUNG MASK
# ══════════════════════════════════════════════════════════════

def _extract_lung_mask(orig_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    soft = cv2.GaussianBlur(filled.astype(np.float32), (51, 51), 0) / 255.0
    return np.clip(soft * 1.2, 0.05, 1.0)


def apply_lung_mask(heatmap: np.ndarray, orig_rgb: np.ndarray) -> np.ndarray:
    H, W   = heatmap.shape
    mask   = _extract_lung_mask(orig_rgb)
    mask_r = cv2.resize(mask, (W, H))
    masked = heatmap * mask_r
    mx     = masked.max()
    if mx > 1e-8:
        masked = masked / mx
    return masked.astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  CLASS-DISCRIMINATIVE GRADCAM++
# ══════════════════════════════════════════════════════════════

class MultiScaleGradCAMPP:
    """
    Class-discriminative multi-scale GradCAM++.

    Computes SEPARATE activation maps for TB class and Normal class,
    then returns the DIFFERENCE map (discriminative map):

        discriminative = TB_cam - Normal_cam  (clipped to [0,1])

    This ensures:
    - TB X-rays show warm activation on lesion regions
    - Normal X-rays show COLD / diffuse activation (no specific hot spots)
    - The heatmaps are visually and clinically distinguishable

    All maps are lung-masked to suppress non-parenchymal activation.
    """

    def __init__(self, model: MultiScaleTBModel):
        self.model = model

    def _single_pass(
        self,
        image: torch.Tensor,
        cls:   int,
        size:  tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward+backward, returns (cam_early, cam_mid, cam_late)."""

        early = self.model.early_attn(self.model.early_backbone(image))
        early.retain_grad()
        mid   = self.model.mid_attn(self.model.mid_backbone(early))
        mid.retain_grad()
        late  = self.model.late_attn(self.model.late_backbone(mid))
        late.retain_grad()

        fused, _ = self.model.fusion([early, mid, late])
        features = self.model.proj(fused)
        logits   = self.model.head(features)

        self.model.zero_grad()
        if image.grad is not None:
            image.grad.zero_()
        logits[0, cls].backward(retain_graph=True)

        def _cam(fmap: torch.Tensor,
                 grad: Optional[torch.Tensor]) -> torch.Tensor:
            if grad is None:
                return torch.zeros(*size, device=fmap.device)
            g2    = grad ** 2
            g3    = grad ** 3
            denom = 2 * g2 + fmap.sum(dim=(2, 3), keepdim=True) * g3
            denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
            alpha   = g2 / denom
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam     = F.relu((weights * fmap).sum(dim=1, keepdim=True))
            cam     = F.interpolate(cam, size=size, mode="bilinear",
                                    align_corners=False).squeeze()
            if cam.max() > 1e-8:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            return cam

        return (
            _cam(early, early.grad),
            _cam(mid,   mid.grad),
            _cam(late,  late.grad),
        )

    @staticmethod
    def _combine(e, m, l) -> torch.Tensor:
        c = 0.2 * e + 0.3 * m + 0.5 * l
        if c.max() > 1e-8:
            c = (c - c.min()) / (c.max() - c.min())
        return c

    def __call__(
        self,
        image:    torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        cls:      int = 1,
        orig_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            primary_cam  : TB-class cam (lung-masked)
            scale_maps   : {
                'tb'            : TB cam (what drives TB prediction)
                'normal'        : Normal cam (what drives Normal prediction)
                'discriminative': TB cam minus Normal cam — the KEY map
                                  Hot on TB lesions, Cold on Normal regions
                'early_tb', 'mid_tb', 'late_tb'
                'early_norm', 'mid_norm', 'late_norm'
            }
        """
        self.model.eval()
        H, W = image.shape[2], image.shape[3]
        size  = (H, W)

        # TB-class forward
        img_tb = image.clone().detach().requires_grad_(True)
        e_tb, m_tb, l_tb = self._single_pass(img_tb, cls=1, size=size)
        cam_tb = self._combine(e_tb, m_tb, l_tb)

        # Normal-class forward
        img_nm = image.clone().detach().requires_grad_(True)
        e_nm, m_nm, l_nm = self._single_pass(img_nm, cls=0, size=size)
        cam_nm = self._combine(e_nm, m_nm, l_nm)

        # Discriminative map: regions SPECIFIC to TB
        disc = torch.clamp(cam_tb - cam_nm, min=0)
        if disc.max() > 1e-8:
            disc = disc / disc.max()

        # Apply lung mask
        def _mask(t: torch.Tensor) -> torch.Tensor:
            if orig_rgb is not None:
                np_cam = apply_lung_mask(t.detach().cpu().numpy(), orig_rgb)
                return torch.from_numpy(np_cam)
            return t.detach()

        cam_tb = _mask(cam_tb)
        cam_nm = _mask(cam_nm)
        disc   = _mask(disc)

        scale_maps = {
            "tb":             cam_tb,
            "normal":         cam_nm,
            "discriminative": disc,       # ← primary map for clinical display
            "early_tb":       e_tb.detach(),
            "mid_tb":         m_tb.detach(),
            "late_tb":        l_tb.detach(),
            "early_norm":     e_nm.detach(),
            "mid_norm":       m_nm.detach(),
            "late_norm":      l_nm.detach(),
        }

        return cam_tb, scale_maps


# ══════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING
# ══════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    def __init__(self, model: MultiScaleTBModel):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, image: torch.Tensor,
                metadata: Optional[torch.Tensor] = None):
        logits, features = self.model(image, metadata)
        return logits / self.temperature, features

    @torch.no_grad()
    def _collect(self, val_loader, device):
        all_logits, all_labels = [], []
        self.model.eval()
        for batch in val_loader:
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(device)
            logits, _ = self.model(imgs, None)
            all_logits.append(logits.cpu())
            if labels.dim() == 2:
                labels = labels.argmax(1)
            all_labels.append(labels.cpu())
        return (torch.cat(all_logits).to(device),
                torch.cat(all_labels).to(device))

    def fit(self, val_loader, device: str = "cpu", max_iter: int = 50):
        self.to(device)
        all_logits, all_labels = self._collect(val_loader, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature],
                                       lr=0.01, max_iter=max_iter)

        def step():
            optimizer.zero_grad()
            loss = criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(step)
        print(f"[TemperatureScaler] T = {self.temperature.item():.4f}")
        return self


# ══════════════════════════════════════════════════════════════
#  TTA
# ══════════════════════════════════════════════════════════════

def predict_with_tta(
    model:    MultiScaleTBModel,
    image:    torch.Tensor,
    n_aug:    int = 8,
    metadata: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    import torchvision.transforms.functional as TF
    model.eval()
    augmentations = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.rotate(x, 5),
        lambda x: TF.rotate(x, -5),
        lambda x: TF.adjust_brightness(x, 1.1),
        lambda x: TF.adjust_brightness(x, 0.9),
        lambda x: TF.adjust_contrast(x, 1.1),
        lambda x: TF.adjust_contrast(x, 0.9),
    ]
    all_probs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for aug in augmentations[:n_aug]:
            aug_img = aug(image.clone()).to(device)
            meta_val = metadata.to(device) if metadata is not None else None
            logits, _ = model(aug_img, meta_val)
            all_probs.append(torch.softmax(logits, dim=1).squeeze().cpu())
    avg = torch.stack(all_probs).mean(0)
    return float(avg[1].item()), float(avg[0].item())


# ══════════════════════════════════════════════════════════════
#  OPTIMAL THRESHOLD
# ══════════════════════════════════════════════════════════════

def find_optimal_threshold(
    model:   MultiScaleTBModel,
    val_loader,
    device:  str = "cpu",
    metric:  str = "youden",
) -> dict:
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs, labels = batch[0], batch[1]
            logits, _ = model(imgs.to(device), None)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs += probs.cpu().tolist()
            if labels.dim() == 2:
                labels = labels.argmax(1)
            all_labels += labels.tolist()

    pa = np.array(all_probs)
    la = np.array(all_labels)
    fpr, tpr, thresholds = roc_curve(la, pa)
    auc = roc_auc_score(la, pa)

    if metric == "youden":
        best_idx = int(np.argmax(tpr + (1 - fpr) - 1))
    else:
        best_idx = int(np.argmax([
            f1_score(la, (pa >= t).astype(int), zero_division=0)
            for t in thresholds
        ]))

    best_t  = float(thresholds[best_idx])
    preds   = (pa >= best_t).astype(int)
    sens    = float(tpr[best_idx])
    spec    = float(1 - fpr[best_idx])
    f1      = float(f1_score(la, preds))
    acc     = float((preds == la).mean())

    print(f"\n{'='*52}")
    print(f"  THRESHOLD ANALYSIS")
    print(f"  AUC         : {auc:.4f}")
    print(f"  Threshold   : {best_t:.4f}  [{metric}]")
    print(f"  Sensitivity : {sens:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"{'='*52}\n")

    return {
        "optimal_threshold": best_t, "auc": auc,
        "sensitivity": sens, "specificity": spec,
        "f1": f1, "accuracy": acc,
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


# ══════════════════════════════════════════════════════════════
#  LEGACY ALIASES
# ══════════════════════════════════════════════════════════════

TBModel = MultiScaleTBModel

class GradCAMPP(MultiScaleGradCAMPP):
    """Legacy alias — keeps old app.py imports working."""
    def __call__(self, image, orig_rgb=None, metadata=None, cls=1):
        cam, _ = super().__call__(image, metadata=metadata,
                                  cls=cls, orig_rgb=orig_rgb)
        return cam


# ══════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════

def freeze_backbone(model: MultiScaleTBModel, freeze: bool = True):
    for part in [model.early_backbone, model.mid_backbone, model.late_backbone]:
        for p in part.parameters():
            p.requires_grad = not freeze

def save_model(model: MultiScaleTBModel, path: str):
    torch.save(model.state_dict(), path)
    print(f"[TBModel] Saved → {path}")

def load_model(path: str, **kwargs) -> MultiScaleTBModel:
    m = MultiScaleTBModel(**kwargs)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m