"""
core/model.py
TB Detection Model — EfficientNetV2-S + SE-CBAM + Metadata Fusion
Includes: GradCAMPP with lung masking + earlier layer hook for better localisation
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  ATTENTION BLOCKS
# ══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        mid = max(ch // r, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class CBAM(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        mid = max(ch // r, 8)
        self.mlp     = nn.Sequential(nn.Linear(ch, mid, bias=False), nn.ReLU(),
                                      nn.Linear(mid, ch, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = self.mlp(x.flatten(2).mean(-1))
        mx  = self.mlp(x.flatten(2).max(-1).values)
        x   = x * torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        s   = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * torch.sigmoid(self.spatial(s))


class SECBAMBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.se   = SEBlock(ch)
        self.cbam = CBAM(ch)
    def forward(self, x): return self.cbam(self.se(x))


# ══════════════════════════════════════════════════════════════
#  METADATA BRANCH
# ══════════════════════════════════════════════════════════════

class MetaBranch(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),  nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════
#  MAIN MODEL
# ══════════════════════════════════════════════════════════════

class TBModel(nn.Module):
    def __init__(
        self,
        num_classes:  int   = 2,
        meta_dim:     int   = 16,
        dropout:      float = 0.4,
        pretrained:   bool  = True,
        use_metadata: bool  = True,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        weights       = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        eff           = efficientnet_v2_s(weights=weights)
        self.backbone = eff.features

        self.attention = SECBAMBlock(1280)
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.proj      = nn.Sequential(
            nn.Linear(1280, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
        )

        self.meta      = MetaBranch(meta_dim, 128) if use_metadata else None
        fusion_dim     = 512 + (128 if use_metadata else 0)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        x        = self.backbone(image)
        x        = self.attention(x)
        x        = self.pool(x).flatten(1)
        features = self.proj(x)

        if self.use_metadata and metadata is not None and self.meta is not None:
            fused = torch.cat([features, self.meta(metadata)], dim=1)
        else:
            fused = features

        return self.head(fused), features


# ══════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.54):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        at = torch.where(targets == 1,
                         torch.full_like(pt, self.alpha),
                         torch.full_like(pt, 1 - self.alpha))
        return (at * (1 - pt) ** self.gamma * ce).mean()


# ══════════════════════════════════════════════════════════════
#  LUNG MASK — suppresses heatmap attention outside lung tissue
# ══════════════════════════════════════════════════════════════

def _extract_lung_mask(orig_rgb: np.ndarray) -> np.ndarray:
    """
    Otsu-based lung mask from RGB X-ray image.
    Returns float32 mask [0,1] same shape as input H×W.
    Lungs are dark on chest X-ray → inverted threshold.
    """
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

    # Otsu threshold — separates dark lung from bright background
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel)

    # Fill holes inside lung regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        # Keep only the two largest contours (left + right lung)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    # Soft edge via blur — smooth transition at lung boundary
    soft = cv2.GaussianBlur(filled.astype(np.float32), (51, 51), 0) / 255.0
    # Boost centre, don't fully zero edges
    soft = np.clip(soft * 1.2, 0.05, 1.0)
    return soft


def apply_lung_mask(heatmap: np.ndarray, orig_rgb: np.ndarray) -> np.ndarray:
    """
    Apply lung mask to GradCAM heatmap.
    heatmap: (H, W) float32 [0,1]
    orig_rgb: (H, W, 3) uint8
    Returns masked + renormalised heatmap.
    """
    H, W   = heatmap.shape
    mask   = _extract_lung_mask(orig_rgb)
    mask_r = cv2.resize(mask, (W, H))
    masked = heatmap * mask_r

    # Renormalise so max=1
    mx = masked.max()
    if mx > 1e-8:
        masked = masked / mx
    return masked.astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  GRADCAM++  — hooks second-to-last block (14×14 spatial)
#               for better TB lesion localisation
# ══════════════════════════════════════════════════════════════

class GradCAMPP:
    """
    GradCAM++ with:
    - Earlier layer hook (-2 instead of -1) → 14×14 vs 7×7 spatial
    - Lung mask applied post-CAM to suppress non-lung attention
    """
    def __init__(self, model: TBModel, layer_idx: int = -2):
        self.model   = model
        self._fmaps  = None
        self._grads  = None
        self._hooks  = []

        # Hook second-to-last backbone block (14×14 feature maps)
        # -1 = last block (7×7, too coarse)
        # -2 = second-to-last (14×14, better for TB lesion localisation)
        target = list(model.backbone.children())[layer_idx]

        self._hooks.append(
            target.register_forward_hook(
                lambda m, i, o: setattr(self, "_fmaps", o)
            )
        )
        self._hooks.append(
            target.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "_grads", go[0])
            )
        )

    def remove_hooks(self):
        for h in self._hooks: h.remove()

    def __call__(
        self,
        image:      torch.Tensor,               # (1, 3, H, W)
        orig_rgb:   Optional[np.ndarray] = None, # (H, W, 3) for lung masking
        metadata:   Optional[torch.Tensor] = None,
        cls:        Optional[int] = None,
    ) -> np.ndarray:                             # returns (H, W) heatmap [0,1]
        self.model.eval()
        self._fmaps = None
        self._grads = None

        img = image.clone().requires_grad_(True)
        logits, _ = self.model(img, metadata)

        cls = cls if cls is not None else int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, cls].backward(retain_graph=False)

        if self._grads is None or self._fmaps is None:
            H, W = image.shape[2:]
            return np.ones((H, W), dtype=np.float32) * 0.5

        g = self._grads.detach()
        f = self._fmaps.detach()

        # GradCAM++ weighting
        g2    = g ** 2
        g3    = g ** 3
        denom = 2 * g2 + (f * g3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        w     = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam   = F.relu((w * f).sum(dim=1, keepdim=True))
        cam   = F.interpolate(cam, image.shape[2:], mode="bilinear", align_corners=False)
        cam   = cam.squeeze().detach().cpu().numpy()

        # Normalise
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min < 1e-8:
            cam = np.zeros_like(cam)
        else:
            cam = (cam - c_min) / (c_max - c_min)

        # Apply lung mask if original image provided
        if orig_rgb is not None:
            cam = apply_lung_mask(cam, orig_rgb)

        return cam


# ══════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING  — calibrates confidence scores
# ══════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    """
    Single learnable temperature parameter T.
    Fit on validation set AFTER training — does not change model weights.
    Usage:
        scaler = TemperatureScaler(model)
        scaler.fit(val_loader)
        calibrated_probs = scaler.predict(image)
    """
    def __init__(self, model: TBModel):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, image: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        logits, features = self.model(image, metadata)
        return logits / self.temperature, features

    def fit(self, val_loader, device: str = "cpu", max_iter: int = 50):
        """Optimise temperature on validation set using NLL loss."""
        self.to(device)
        self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        # Collect all logits and labels first
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                logits, _ = self.model(imgs, None)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"[TemperatureScaler] Fitted T = {self.temperature.item():.4f}")
        return self


# ══════════════════════════════════════════════════════════════
#  TTA — Test Time Augmentation
# ══════════════════════════════════════════════════════════════

def predict_with_tta(
    model:    TBModel,
    image:    torch.Tensor,          # (1, 3, H, W) — normalised
    n_aug:    int = 8,
    metadata: Optional[torch.Tensor] = None,
) -> tuple[float, float]:
    """
    Run n_aug augmented versions of the image through the model.
    Average probabilities → more stable, +1-3% accuracy.
    Returns (tb_prob, normal_prob).
    """
    import torchvision.transforms.functional as TF

    model.eval()
    all_probs = []

    augmentations = [
        lambda x: x,                                           # original
        lambda x: TF.hflip(x),                                # horizontal flip
        lambda x: TF.rotate(x, 5),                            # rotate +5°
        lambda x: TF.rotate(x, -5),                           # rotate -5°
        lambda x: TF.adjust_brightness(x, 1.1),               # brighter
        lambda x: TF.adjust_brightness(x, 0.9),               # darker
        lambda x: TF.adjust_contrast(x, 1.1),                 # more contrast
        lambda x: TF.adjust_contrast(x, 0.9),                 # less contrast
    ]

    with torch.no_grad():
        for aug in augmentations[:n_aug]:
            aug_img = aug(image.clone())
            logits, _ = model(aug_img, metadata)
            probs = torch.softmax(logits, dim=1).squeeze()
            all_probs.append(probs.cpu())

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return float(avg_probs[1].item()), float(avg_probs[0].item())


# ══════════════════════════════════════════════════════════════
#  OPTIMAL THRESHOLD FINDER
# ══════════════════════════════════════════════════════════════

def find_optimal_threshold(
    model:      TBModel,
    val_loader,
    device:     str = "cpu",
    metric:     str = "youden",   # "youden" | "f1"
) -> dict:
    """
    Compute ROC curve on validation set.
    Find threshold that maximises Youden's J (sensitivity + specificity - 1)
    or F1 score — both better than fixed 0.5 for clinical use.
    Returns dict with optimal_threshold, sensitivity, specificity, f1, auc.
    """
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score
    import numpy as np

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs    = imgs.to(device)
            logits, _ = model(imgs, None)
            probs   = torch.softmax(logits, dim=1)[:, 1]
            all_probs  += probs.cpu().tolist()
            all_labels += labels.tolist()

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)

    fpr, tpr, thresholds = roc_curve(labels_arr, probs_arr)
    auc = roc_auc_score(labels_arr, probs_arr)

    if metric == "youden":
        # Youden's J = sensitivity + specificity - 1
        specificity = 1 - fpr
        j_scores    = tpr + specificity - 1
        best_idx    = np.argmax(j_scores)
    else:
        # F1 score at each threshold
        f1_scores = []
        for t in thresholds:
            preds = (probs_arr >= t).astype(int)
            f1_scores.append(f1_score(labels_arr, preds, zero_division=0))
        best_idx = np.argmax(f1_scores)

    best_t    = float(thresholds[best_idx])
    best_preds = (probs_arr >= best_t).astype(int)
    sens       = float(tpr[best_idx])
    spec       = float(1 - fpr[best_idx])
    f1         = float(f1_score(labels_arr, best_preds))
    acc        = float((best_preds == labels_arr).mean())

    print(f"\n{'='*50}")
    print(f"OPTIMAL THRESHOLD ANALYSIS")
    print(f"{'='*50}")
    print(f"  AUC-ROC:       {auc:.4f}")
    print(f"  Best Threshold:{best_t:.4f}  (metric: {metric})")
    print(f"  Sensitivity:   {sens:.4f}  (TB recall — catching real TB cases)")
    print(f"  Specificity:   {spec:.4f}  (avoiding false alarms)")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"{'='*50}\n")

    return {
        "optimal_threshold": best_t,
        "auc":               auc,
        "sensitivity":       sens,
        "specificity":       spec,
        "f1":                f1,
        "accuracy":          acc,
        "fpr":               fpr.tolist(),
        "tpr":               tpr.tolist(),
        "thresholds":        thresholds.tolist(),
    }


# ══════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════

def freeze_backbone(model: TBModel, freeze: bool = True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze

def save_model(model: TBModel, path: str):
    torch.save(model.state_dict(), path)
    print(f"[TBModel] Saved → {path}")

def load_model(path: str, **kwargs) -> TBModel:
    m = TBModel(**kwargs)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m
# ═══════════════════════════════════════════════════════════════
#  MULTI-SCALE FEATURE FUSION — Scale Fusion + MultiScaleTBModel
# ═══════════════════════════════════════════════════════════════

class ScaleFusion(nn.Module):
    """Projects each scale's features to a common dim and fuses them."""
    def __init__(self, in_dims, out_dim=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )
            for d in in_dims
        ])
        self.scale_attn = nn.Sequential(
            nn.Linear(out_dim * len(in_dims), len(in_dims)),
            nn.Softmax(dim=-1),
        )

    def forward(self, feature_maps):
        projected = [proj(fm) for proj, fm in zip(self.projs, feature_maps)]
        concat    = torch.cat(projected, dim=-1)
        weights   = self.scale_attn(concat)          # (B, num_scales)
        stacked   = torch.stack(projected, dim=1)    # (B, num_scales, out_dim)
        weights   = weights.unsqueeze(-1)             # (B, num_scales, 1)
        fused     = (stacked * weights).sum(dim=1)   # (B, out_dim)
        return fused, weights.squeeze(-1)


class MultiScaleTBModel(nn.Module):
    """
    EfficientNetV2-S with multi-scale feature extraction.

    Backbone stages tapped:
      Stage 2 (early)  — 48ch,  28x28  — detects small nodules
      Stage 5 (mid)    — 160ch, 14x14  — detects consolidation
      Stage 7 (late)   — 1280ch, 7x7   — detects large cavities
    """

    def __init__(
        self,
        num_classes:  int   = 2,
        pretrained:   bool  = True,
        dropout:      float = 0.4,
        fusion_dim:   int   = 256,
        use_metadata: bool  = False,
        meta_dim:     int   = 16,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        eff     = efficientnet_v2_s(weights=weights)
        blocks  = list(eff.features.children())   # 9 blocks total (0–8)

        # Split backbone into 3 scale segments
        self.early_backbone = nn.Sequential(*blocks[:3])   # → 48ch,  28x28
        self.mid_backbone   = nn.Sequential(*blocks[3:6])  # → 160ch, 14x14
        self.late_backbone  = nn.Sequential(*blocks[6:])   # → 1280ch, 7x7

        # Attention at each scale
        self.early_attn = SECBAMBlock(48)
        self.mid_attn   = SECBAMBlock(160)
        self.late_attn  = SECBAMBlock(1280)

        # Multi-scale fusion
        self.fusion = ScaleFusion(in_dims=[48, 160, 1280], out_dim=fusion_dim)

        # Project to 512-dim (keeps DR model compatibility)
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Optional metadata branch
        if use_metadata:
            self.meta_branch = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            head_in = 512 + 128
        else:
            self.meta_branch = None
            head_in = 512

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata=None):
        early = self.early_attn(self.early_backbone(image))   # (B, 48,  28, 28)
        mid   = self.mid_attn(self.mid_backbone(early))       # (B, 160, 14, 14)
        late  = self.late_attn(self.late_backbone(mid))       # (B, 1280, 7,  7)

        fused, scale_weights = self.fusion([early, mid, late])
        features = self.proj(fused)   # (B, 512)

        if self.use_metadata and metadata is not None and self.meta_branch is not None:
            combined = torch.cat([features, self.meta_branch(metadata)], dim=1)
        else:
            combined = features

        return self.head(combined), features   # same interface as TBModel


class MultiScaleGradCAMPP:
    """
    GradCAM++ across all 3 scales simultaneously.
    Returns combined heatmap + individual scale maps.
    """

    def __init__(self, model: MultiScaleTBModel):
        self.model   = model
        self.maps_e  = {};  self.grads_e = {}
        self.maps_m  = {};  self.grads_m = {}
        self.maps_l  = {};  self.grads_l = {}
        self._register_hooks()

    def _register_hooks(self):
        def save_fmap(store, key):
            def hook(m, inp, out): store[key] = out
            return hook

    def save_grad(store, key):
        def hook(m, inp, out):
            if out.requires_grad:          # only register if gradients are on
                def bk(g): store[key] = g
                out.register_hook(bk)
        return hook

        last_early = list(self.model.early_backbone.children())[-1]
        last_mid   = list(self.model.mid_backbone.children())[-1]
        last_late  = list(self.model.late_backbone.children())[-1]

        last_early.register_forward_hook(save_fmap(self.maps_e, "e"))
        last_early.register_forward_hook(save_grad(self.grads_e, "e"))
        last_mid.register_forward_hook(save_fmap(self.maps_m, "m"))
        last_mid.register_forward_hook(save_grad(self.grads_m, "m"))
        last_late.register_forward_hook(save_fmap(self.maps_l, "l"))
        last_late.register_forward_hook(save_grad(self.grads_l, "l"))

    def _compute_cam(self, fmap, grad):
        g2    = grad ** 2
        g3    = grad ** 3
        denom = 2 * g2 + fmap.sum(dim=(2, 3), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha   = g2 / denom
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * fmap).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(224, 224),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze()
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def __call__(self, image: torch.Tensor, metadata=None, cls: int = 1):
        self.model.eval()

        # Need gradients on image and all params temporarily
        image = image.requires_grad_(True)

        # Manual forward capturing intermediate outputs
        early_out = self.model.early_attn(self.model.early_backbone(image))
        early_out.retain_grad()

        mid_out   = self.model.mid_attn(self.model.mid_backbone(early_out))
        mid_out.retain_grad()

        late_out  = self.model.late_attn(self.model.late_backbone(mid_out))
        late_out.retain_grad()

        fused, scale_weights = self.model.fusion([early_out, mid_out, late_out])
        features  = self.model.proj(fused)
        logits    = self.model.head(features)

        self.model.zero_grad()
        logits[0, cls].backward()

        def cam_from(fmap, grad):
            if grad is None:
                return torch.zeros(224, 224)
            g2    = grad ** 2
            g3    = grad ** 3
            denom = 2 * g2 + fmap.sum(dim=(2,3), keepdim=True) * g3
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha   = g2 / denom
            weights = (alpha * F.relu(grad)).sum(dim=(2,3), keepdim=True)
            cam     = F.relu((weights * fmap).sum(dim=1, keepdim=True))
            cam     = F.interpolate(cam, size=(224,224), mode="bilinear", align_corners=False)
            cam     = cam.squeeze()
            if cam.max() > 1e-8:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            return cam

        cam_e = cam_from(early_out, early_out.grad)
        cam_m = cam_from(mid_out,   mid_out.grad)
        cam_l = cam_from(late_out,  late_out.grad)

        combined = 0.2 * cam_e + 0.3 * cam_m + 0.5 * cam_l
        if combined.max() > 1e-8:
            combined = (combined - combined.min()) / (combined.max() - combined.min())

        return combined, {"early": cam_e, "mid": cam_m, "late": cam_l}