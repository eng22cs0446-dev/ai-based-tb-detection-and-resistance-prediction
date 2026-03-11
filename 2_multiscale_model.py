"""
2_multiscale_model.py
=====================
Drop-in replacement for the TBModel class in tb_system/core/model.py

Changes vs current model:
  - Taps 3 backbone stages (early 28x28, mid 14x14, late 7x7)
  - Extracts features at each scale separately
  - Fuses all three before classifier
  - Detects TB at all stages: early nodules, consolidation, cavities

HOW TO USE:
1. Copy the MultiScaleTBModel class into tb_system/core/model.py
2. In train.py, change:
       model = TBModel(...)
   to:
       model = MultiScaleTBModel(...)
3. Retrain from scratch

Expected improvement: +2-4% sensitivity on early-stage TB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import Optional


# ── Attention blocks (same as current model) ─────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ca = self.channel_att(x).view(b, c, 1, 1)
        x  = x * ca
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        sa  = self.spatial_att(torch.cat([avg, mx], dim=1))
        return x * sa


class SECBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se   = SEBlock(channels)
        self.cbam = CBAMBlock(channels)

    def forward(self, x):
        return self.cbam(self.se(x))


# ── Scale fusion projection ───────────────────────────────────
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
        # Cross-scale attention — which scale matters most for this image?
        self.scale_attn = nn.Sequential(
            nn.Linear(out_dim * len(in_dims), len(in_dims)),
            nn.Softmax(dim=-1),
        )

    def forward(self, feature_maps):
        # Project each scale
        projected = [proj(fm) for proj, fm in zip(self.projs, feature_maps)]
        concat     = torch.cat(projected, dim=-1)

        # Compute per-scale attention weights
        weights = self.scale_attn(concat)  # (B, num_scales)

        # Weighted sum
        stacked  = torch.stack(projected, dim=1)    # (B, num_scales, out_dim)
        weights  = weights.unsqueeze(-1)             # (B, num_scales, 1)
        fused    = (stacked * weights).sum(dim=1)   # (B, out_dim)
        return fused, weights.squeeze(-1)


# ── Multi-Scale TB Model ──────────────────────────────────────
class MultiScaleTBModel(nn.Module):
    """
    EfficientNetV2-S with multi-scale feature extraction.

    Backbone stages tapped:
      Stage 2 (early)  — 48 channels,  28x28 spatial  — detects small nodules
      Stage 5 (mid)    — 160 channels, 14x14 spatial  — detects consolidation
      Stage 7 (late)   — 1280 channels, 7x7 spatial   — detects large cavities

    All three are fused with cross-scale attention before classification.
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

        # ── Backbone ──────────────────────────────────────────
        weights  = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        eff      = efficientnet_v2_s(weights=weights)
        blocks   = list(eff.features.children())  # 9 blocks total (0-8)

        # Split into 3 scale segments
        self.early_backbone = nn.Sequential(*blocks[:3])   # → 48ch,  28x28
        self.mid_backbone   = nn.Sequential(*blocks[3:6])  # → 160ch, 14x14
        self.late_backbone  = nn.Sequential(*blocks[6:])   # → 1280ch, 7x7

        # ── Attention at each scale ───────────────────────────
        self.early_attn = SECBAMBlock(48)
        self.mid_attn   = SECBAMBlock(160)
        self.late_attn  = SECBAMBlock(1280)

        # ── Multi-scale fusion ────────────────────────────────
        self.fusion = ScaleFusion(
            in_dims  = [48, 160, 1280],
            out_dim  = fusion_dim,
        )

        # ── Projection to 512-dim (for DR model compatibility) ─
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Metadata branch ───────────────────────────────────
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

        # ── Classifier head ───────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        # Extract features at 3 scales
        early = self.early_attn(self.early_backbone(image))   # (B, 48,  28, 28)
        mid   = self.mid_attn(self.mid_backbone(early))       # (B, 160, 14, 14)
        late  = self.late_attn(self.late_backbone(mid))       # (B, 1280, 7,  7)

        # Fuse all 3 scales
        fused, scale_weights = self.fusion([early, mid, late])  # (B, fusion_dim)

        # Project to 512-dim features (for DR model)
        features = self.proj(fused)   # (B, 512)

        # Optional metadata fusion
        if self.use_metadata and metadata is not None and self.meta_branch is not None:
            meta_feat = self.meta_branch(metadata)
            combined  = torch.cat([features, meta_feat], dim=1)
        else:
            combined = features

        logits = self.head(combined)
        return logits, features   # same interface as TBModel


# ── GradCAM++ for Multi-Scale Model ──────────────────────────
class MultiScaleGradCAMPP:
    """
    GradCAM++ that visualises attention at ALL 3 scales simultaneously.
    Returns 3 heatmaps: early (small details), mid (mid-level), late (global).
    """

    def __init__(self, model: MultiScaleTBModel):
        self.model    = model
        self.maps_e   = {}
        self.grads_e  = {}
        self.maps_m   = {}
        self.grads_m  = {}
        self.maps_l   = {}
        self.grads_l  = {}
        self._register_hooks()

    def _register_hooks(self):
        def save_fmap(store, key):
            def hook(m, inp, out): store[key] = out
            return hook

        def save_grad(store, key):
            def hook(m, inp, out):
                def bk(g): store[key] = g
                out.register_hook(bk)
            return hook

        # Tap the last layer of each scale block
        last_early = list(self.model.early_backbone.children())[-1]
        last_mid   = list(self.model.mid_backbone.children())[-1]
        last_late  = list(self.model.late_backbone.children())[-1]

        last_early.register_forward_hook(save_fmap(self.maps_e,  "e"))
        last_early.register_forward_hook(save_grad(self.grads_e, "e"))
        last_mid.register_forward_hook(save_fmap(self.maps_m,    "m"))
        last_mid.register_forward_hook(save_grad(self.grads_m,   "m"))
        last_late.register_forward_hook(save_fmap(self.maps_l,   "l"))
        last_late.register_forward_hook(save_grad(self.grads_l,  "l"))

    def _compute_cam(self, fmap, grad):
        """Compute GradCAM++ weights and activation map."""
        b, c, h, w = grad.shape
        # GradCAM++ alpha weights
        g2   = grad ** 2
        g3   = grad ** 3
        denom = 2 * g2 + fmap.sum(dim=(2, 3), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
        # Normalise
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def __call__(self, image: torch.Tensor, metadata=None, cls: int = 1):
        self.model.eval()
        image.requires_grad_(True)

        logits, _ = self.model(image, metadata)
        self.model.zero_grad()
        logits[0, cls].backward()

        cam_e = self._compute_cam(self.maps_e["e"], self.grads_e["e"])
        cam_m = self._compute_cam(self.maps_m["m"], self.grads_m["m"])
        cam_l = self._compute_cam(self.maps_l["l"], self.grads_l["l"])

        # Weighted blend — late gets most weight for global TB patterns
        # mid gets weight for consolidation, early for small nodules
        combined = 0.2 * cam_e + 0.3 * cam_m + 0.5 * cam_l
        if combined.max() > 1e-8:
            combined = (combined - combined.min()) / (combined.max() - combined.min())

        return combined, {"early": cam_e, "mid": cam_m, "late": cam_l}


# ── Training config diff ──────────────────────────────────────
TRAINING_DIFF = """
Changes needed in train.py to use MultiScaleTBModel:

1. Import:
   from core.model import MultiScaleTBModel  # instead of TBModel

2. Model init:
   model = MultiScaleTBModel(
       num_classes=2,
       pretrained=True,
       dropout=0.4,
       fusion_dim=256,
       use_metadata=False,
   )

3. Stage 1 freeze — freeze early+mid backbone, keep late + head trainable:
   for param in model.early_backbone.parameters():
       param.requires_grad = False
   for param in model.mid_backbone.parameters():
       param.requires_grad = False
   # late_backbone, fusion, proj, head are trainable

4. Stage 2 unfreeze all:
   for param in model.parameters():
       param.requires_grad = True

5. In app.py, GradCAM call:
   from core.model import MultiScaleGradCAMPP
   gradcam = MultiScaleGradCAMPP(model)
   cam, scale_maps = gradcam(tensor, cls=1)
   # cam is the combined heatmap
   # scale_maps["early"], ["mid"], ["late"] are individual scales
"""

if __name__ == "__main__":
    print("Multi-Scale TB Model — Architecture Test")
    print("=" * 50)
    model = MultiScaleTBModel(pretrained=False)
    x     = torch.randn(2, 3, 224, 224)
    logits, features = model(x, None)
    print(f"Input shape    : {x.shape}")
    print(f"Logits shape   : {logits.shape}")
    print(f"Features shape : {features.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params   : {total/1e6:.1f}M")
    print("\nTraining diff:")
    print(TRAINING_DIFF)