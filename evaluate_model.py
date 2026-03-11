"""
evaluate_model.py
=================
Run this from your project root:
    python evaluate_model.py

Generates a full evaluation report with:
  - Confusion Matrix
  - ROC Curve (with AUC)
  - Precision-Recall Curve (with AP)
  - F1 / Precision / Recall at multiple thresholds
  - Per-class metrics table
  - Probability distribution histogram
  - Saves everything to:  results/evaluation_report.png
                          results/metrics_summary.txt
"""

import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── auto-find project root ──────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "tb_system")):
        break
    _root = os.path.dirname(_root)

sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tb_system"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)

from core.model         import TBModel, TemperatureScaler
from core.preprocessing import TBDataset

# ── PATHS ───────────────────────────────────────────────────
CKPT_DIR  = os.path.join(_root, "checkpoints")
DATA_ROOT = os.path.join(_root, "data")
OUT_DIR   = os.path.join(_root, "results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── COLOURS (dark theme matching the app) ───────────────────
BG    = "#080D14"
PANEL = "#0D1623"
BLUE  = "#3B8FE8"
TEAL  = "#0E9E8E"
GREEN = "#18A558"
RED   = "#D93025"
ORG   = "#E8650A"
YEL   = "#D4A017"
WHITE = "#D8E4F5"
GREY  = "#4A6380"

# ─────────────────────────────────────────────────────────────
print("\n" + "="*58)
print("  TB Detection System — Full Evaluation")
print("="*58)

# ── 1. LOAD MODEL ────────────────────────────────────────────
print(f"\n[1/5] Loading model from {CKPT_DIR}...")
ckpt = os.path.join(CKPT_DIR, "best_model.pth")
if not os.path.exists(ckpt):
    print(f"  ERROR: {ckpt} not found"); sys.exit(1)

model = TBModel(num_classes=2, pretrained=False, use_metadata=False)
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.to(DEVICE).eval()

# Load temperature scaler if available
t_path = os.path.join(CKPT_DIR, "temperature_scaler.pth")
scaler = None
if os.path.exists(t_path):
    scaler = TemperatureScaler(model)
    scaler.load_state_dict(torch.load(t_path, map_location=DEVICE))
    scaler.to(DEVICE).eval()
    T_val = scaler.temperature.item()
    print(f"  Temperature scaler loaded  (T = {T_val:.4f})")
else:
    print("  No temperature scaler found — using raw probabilities")
    T_val = None

# Load threshold
thresh_path = os.path.join(CKPT_DIR, "optimal_threshold.json")
threshold = 0.5
if os.path.exists(thresh_path):
    with open(thresh_path) as f:
        threshold = json.load(f)["optimal_threshold"]
print(f"  Optimal threshold: {threshold:.4f}")

# ── 2. LOAD VALIDATION DATA ──────────────────────────────────
print(f"\n[2/5] Loading validation set from {DATA_ROOT}...")
val_ds = TBDataset(DATA_ROOT, split="val", size=224)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=(DEVICE=="cuda"))
print(f"  {len(val_ds)} images  (Normal + TB)")

# ── 3. RUN INFERENCE ─────────────────────────────────────────
print(f"\n[3/5] Running inference on validation set...")
t0 = time.time()
all_probs, all_labels = [], []

infer_model = scaler if scaler else model
with torch.no_grad():
    for i, (imgs, labels, *_) in enumerate(val_loader):
        imgs = imgs.to(DEVICE)
        logits, _ = infer_model(imgs, None)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(TB)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        if (i+1) % 5 == 0:
            print(f"    Batch {i+1}/{len(val_loader)}", end="\r")

elapsed = time.time() - t0
all_probs  = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_preds  = (all_probs >= threshold).astype(int)

print(f"\n  Done in {elapsed:.1f}s  ({len(all_probs)} samples)")

# ── 4. COMPUTE METRICS ───────────────────────────────────────
print(f"\n[4/5] Computing metrics...")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
TN, FP, FN, TP = cm.ravel()

# Core metrics
accuracy    = accuracy_score(all_labels, all_preds)
sensitivity = recall_score(all_labels, all_preds, pos_label=1)    # TPR
specificity = recall_score(all_labels, all_preds, pos_label=0)    # TNR
precision   = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
f1          = f1_score(all_labels, all_preds, pos_label=1)
ppv         = precision  # Positive Predictive Value = Precision
npv         = TN / (TN + FN) if (TN + FN) > 0 else 0

# ROC
fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Precision-Recall
prec_curve, rec_curve, pr_thresholds = precision_recall_curve(all_labels, all_probs)
avg_precision = average_precision_score(all_labels, all_probs)

# F1 at multiple thresholds
thresh_range = np.linspace(0.1, 0.9, 81)
f1_scores_t, prec_t, rec_t = [], [], []
for t in thresh_range:
    p = (all_probs >= t).astype(int)
    f1_scores_t.append(f1_score(all_labels, p, zero_division=0))
    prec_t.append(precision_score(all_labels, p, pos_label=1, zero_division=0))
    rec_t.append(recall_score(all_labels, p, pos_label=1, zero_division=0))

best_f1_thresh = thresh_range[np.argmax(f1_scores_t)]
best_f1_val    = max(f1_scores_t)

print(f"  AUC-ROC     : {roc_auc:.4f}")
print(f"  F1 Score    : {f1:.4f}  (best F1 {best_f1_val:.4f} at thresh {best_f1_thresh:.4f})")
print(f"  Sensitivity : {sensitivity:.4f}  (Recall / TPR)")
print(f"  Specificity : {specificity:.4f}  (TNR)")
print(f"  Precision   : {precision:.4f}  (PPV)")
print(f"  Accuracy    : {accuracy:.4f}")
print(f"  NPV         : {npv:.4f}")
print(f"  Avg Prec    : {avg_precision:.4f}  (PR-AUC)")
print(f"\n  Confusion Matrix:")
print(f"              Pred Normal  Pred TB")
print(f"  True Normal    {TN:5d}    {FP:5d}")
print(f"  True TB        {FN:5d}    {TP:5d}")

# ── 5. PLOT ──────────────────────────────────────────────────
print(f"\n[5/5] Generating plots...")

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle("TB Detection System — Model Evaluation Report",
             fontsize=22, fontweight="bold", color=WHITE, y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38,
                       left=0.05, right=0.97, top=0.93, bottom=0.06)

# ── helper ────────────────────────────────────────────────────
def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────
# ROW 0: Metric cards (spanning all 4 cols)
# ─────────────────────────────────────────────────────────────
card_ax = fig.add_subplot(gs[0, :])
card_ax.set_facecolor(BG)
card_ax.axis("off")

metrics_cards = [
    (f"{roc_auc:.4f}", "AUC-ROC",     BLUE),
    (f"{f1:.4f}",      "F1 Score",    TEAL),
    (f"{sensitivity:.1%}", "Sensitivity\n(Recall)", ORG),
    (f"{specificity:.1%}", "Specificity\n(TNR)",   GREEN),
    (f"{precision:.1%}",   "Precision\n(PPV)",     YEL),
    (f"{accuracy:.1%}",    "Accuracy",             BLUE),
    (f"{npv:.1%}",         "NPV",                  TEAL),
    (f"{avg_precision:.4f}","PR-AUC\n(Avg Prec)",  ORG),
]

n = len(metrics_cards)
for i, (val, lbl, col) in enumerate(metrics_cards):
    x0 = i / n
    fancy = FancyBboxPatch((x0 + 0.005, 0.05), 1/n - 0.012, 0.88,
                           boxstyle="round,pad=0.01",
                           facecolor=PANEL, edgecolor=col, linewidth=1.5,
                           transform=card_ax.transAxes)
    card_ax.add_patch(fancy)
    card_ax.text(x0 + 1/(2*n), 0.7, val,
                 ha="center", va="center", fontsize=18, fontweight="bold",
                 color=col, transform=card_ax.transAxes)
    card_ax.text(x0 + 1/(2*n), 0.22, lbl,
                 ha="center", va="center", fontsize=8.5,
                 color=GREY, transform=card_ax.transAxes, linespacing=1.3)

# ─────────────────────────────────────────────────────────────
# ROW 1, COL 0-1: Confusion Matrix
# ─────────────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1, 0:2])
style_ax(ax_cm, "Confusion Matrix")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
colours_cm = [[GREEN if i == j else RED for j in range(2)] for i in range(2)]
cell_colors_alpha = [[0.35 if i == j else 0.2 for j in range(2)] for i in range(2)]

for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct   = cm_norm[i, j] * 100
        col   = GREEN if i == j else RED
        alpha = 0.4 if i == j else 0.2
        ax_cm.add_patch(FancyBboxPatch((j+0.03, i+0.03), 0.94, 0.94,
                                       boxstyle="round,pad=0.02",
                                       facecolor=col, alpha=alpha,
                                       edgecolor=col, linewidth=2))
        ax_cm.text(j+0.5, i+0.62, str(count),
                   ha="center", va="center", fontsize=26,
                   fontweight="bold", color=col)
        ax_cm.text(j+0.5, i+0.28, f"{pct:.1f}%",
                   ha="center", va="center", fontsize=13, color=WHITE)

labels_cm = ["Normal", "TB"]
ax_cm.set_xticks([0.5, 1.5])
ax_cm.set_yticks([0.5, 1.5])
ax_cm.set_xticklabels(["Pred: Normal", "Pred: TB"], color=WHITE, fontsize=10)
ax_cm.set_yticklabels(["True: Normal", "True: TB"], color=WHITE, fontsize=10)
ax_cm.set_xlim(0, 2); ax_cm.set_ylim(0, 2)

# labels
ax_cm.text(0.5, 2.15, f"TN={TN}  FP={FP}", ha="center", color=GREY, fontsize=9,
           transform=ax_cm.get_xaxis_transform())
ax_cm.text(-0.18, 0.5, f"FN={FN}", ha="center", color=RED, fontsize=9,
           transform=ax_cm.transData, rotation=90, va="center")
ax_cm.text(-0.18, 1.5, f"TP={TP}", ha="center", color=GREEN, fontsize=9,
           transform=ax_cm.transData, rotation=90, va="center")

# ─────────────────────────────────────────────────────────────
# ROW 1, COL 2-3: ROC Curve
# ─────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 2:4])
style_ax(ax_roc, f"ROC Curve  —  AUC = {roc_auc:.4f}")

ax_roc.plot([0,1],[0,1], "--", color=GREY, linewidth=1, alpha=0.5, label="Random (AUC=0.50)")
ax_roc.plot(fpr, tpr, color=BLUE, linewidth=2.5, label=f"TB Model  (AUC = {roc_auc:.4f})")
ax_roc.fill_between(fpr, tpr, alpha=0.12, color=BLUE)

# mark operating point at chosen threshold
op_fpr = 1 - specificity
op_tpr = sensitivity
ax_roc.scatter([op_fpr], [op_tpr], s=120, color=ORG, zorder=5,
               label=f"Operating point (thresh={threshold:.4f})")
ax_roc.annotate(f"Sens {sensitivity:.1%}\nSpec {specificity:.1%}",
                xy=(op_fpr, op_tpr), xytext=(op_fpr+0.12, op_tpr-0.12),
                color=ORG, fontsize=8.5, arrowprops=dict(arrowstyle="->", color=ORG))

ax_roc.set_xlabel("False Positive Rate  (1 - Specificity)")
ax_roc.set_ylabel("True Positive Rate  (Sensitivity)")
ax_roc.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=8.5)
ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.05)
ax_roc.grid(True, color=GREY, alpha=0.2, linewidth=0.4)

# ─────────────────────────────────────────────────────────────
# ROW 2, COL 0-1: Precision-Recall Curve
# ─────────────────────────────────────────────────────────────
ax_pr = fig.add_subplot(gs[2, 0:2])
style_ax(ax_pr, f"Precision-Recall Curve  —  AP = {avg_precision:.4f}")

baseline = all_labels.mean()
ax_pr.axhline(y=baseline, color=GREY, linestyle="--", linewidth=1, alpha=0.5,
              label=f"Baseline (AP={baseline:.2f})")
ax_pr.plot(rec_curve, prec_curve, color=TEAL, linewidth=2.5,
           label=f"TB Model  (AP = {avg_precision:.4f})")
ax_pr.fill_between(rec_curve, prec_curve, alpha=0.12, color=TEAL)

# mark operating point
ax_pr.scatter([sensitivity], [precision], s=120, color=ORG, zorder=5,
              label=f"Operating point (thresh={threshold:.4f})")
ax_pr.annotate(f"Prec {precision:.1%}\nRecall {sensitivity:.1%}",
               xy=(sensitivity, precision),
               xytext=(sensitivity-0.18, precision+0.08),
               color=ORG, fontsize=8.5, arrowprops=dict(arrowstyle="->", color=ORG))

ax_pr.set_xlabel("Recall  (Sensitivity)")
ax_pr.set_ylabel("Precision  (PPV)")
ax_pr.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=8.5)
ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(-0.02, 1.05)
ax_pr.grid(True, color=GREY, alpha=0.2, linewidth=0.4)

# ─────────────────────────────────────────────────────────────
# ROW 2, COL 2: F1 / Precision / Recall vs Threshold
# ─────────────────────────────────────────────────────────────
ax_f1 = fig.add_subplot(gs[2, 2])
style_ax(ax_f1, "F1 / Precision / Recall vs Threshold")

ax_f1.plot(thresh_range, f1_scores_t, color=TEAL,  linewidth=2,   label="F1")
ax_f1.plot(thresh_range, prec_t,      color=YEL,   linewidth=1.5, label="Precision", linestyle="--")
ax_f1.plot(thresh_range, rec_t,       color=ORG,   linewidth=1.5, label="Recall",    linestyle="-.")

ax_f1.axvline(x=threshold, color=WHITE, linewidth=1.2, linestyle=":",
              label=f"Current threshold {threshold:.2f}")
ax_f1.axvline(x=best_f1_thresh, color=TEAL, linewidth=1, linestyle=":",
              alpha=0.5, label=f"Best F1 thresh {best_f1_thresh:.2f}")
ax_f1.scatter([threshold], [f1], color=WHITE, s=60, zorder=5)

ax_f1.set_xlabel("Threshold")
ax_f1.set_ylabel("Score")
ax_f1.set_xlim(0.1, 0.9); ax_f1.set_ylim(0, 1.05)
ax_f1.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=7.5)
ax_f1.grid(True, color=GREY, alpha=0.2, linewidth=0.4)

# ─────────────────────────────────────────────────────────────
# ROW 2, COL 3: Probability Distribution
# ─────────────────────────────────────────────────────────────
ax_hist = fig.add_subplot(gs[2, 3])
style_ax(ax_hist, "TB Probability Distribution")

normal_probs = all_probs[all_labels == 0]
tb_probs     = all_probs[all_labels == 1]

ax_hist.hist(normal_probs, bins=40, color=GREEN, alpha=0.6, label=f"True Normal (n={len(normal_probs)})", density=True)
ax_hist.hist(tb_probs,     bins=40, color=RED,   alpha=0.6, label=f"True TB (n={len(tb_probs)})", density=True)
ax_hist.axvline(x=threshold, color=WHITE, linewidth=1.5, linestyle="--",
                label=f"Threshold {threshold:.4f}")
ax_hist.axvspan(0.0,       0.30,      alpha=0.06, color=GREEN)
ax_hist.axvspan(0.30,      threshold, alpha=0.06, color=YEL)
ax_hist.axvspan(threshold, 1.0,       alpha=0.06, color=RED)

ax_hist.text(0.15, ax_hist.get_ylim()[1]*0.85 if ax_hist.get_ylim()[1] > 0 else 3,
             "Normal\nzone", ha="center", color=GREEN, fontsize=7.5)
ax_hist.text(threshold+0.12, 0.5, "TB\nzone", ha="center", color=RED, fontsize=7.5)

ax_hist.set_xlabel("P(TB)")
ax_hist.set_ylabel("Density")
ax_hist.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=7.5)
ax_hist.set_xlim(-0.02, 1.02)
ax_hist.grid(True, color=GREY, alpha=0.2, linewidth=0.4)

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
out_png = os.path.join(OUT_DIR, "evaluation_report.png")
fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\n  Saved -> {out_png}")

# ── SAVE TEXT SUMMARY ────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "metrics_summary.txt")
with open(summary_path, "w") as f:
    f.write("TB DETECTION SYSTEM — MODEL EVALUATION SUMMARY\n")
    f.write("=" * 54 + "\n\n")
    f.write(f"Model       : {ckpt}\n")
    f.write(f"Device      : {DEVICE}\n")
    f.write(f"Threshold   : {threshold:.4f}  (Youden's J optimal)\n")
    f.write(f"Calibrated  : {'Yes, T=' + str(round(T_val,4)) if T_val else 'No'}\n")
    f.write(f"Dataset     : Validation set, {len(val_ds)} images\n\n")
    f.write("CORE METRICS\n" + "-"*40 + "\n")
    f.write(f"AUC-ROC         : {roc_auc:.4f}\n")
    f.write(f"PR-AUC (Avg P)  : {avg_precision:.4f}\n")
    f.write(f"F1 Score        : {f1:.4f}\n")
    f.write(f"  Best F1       : {best_f1_val:.4f}  at threshold {best_f1_thresh:.4f}\n")
    f.write(f"Accuracy        : {accuracy:.4f}  ({accuracy:.1%})\n")
    f.write(f"Sensitivity     : {sensitivity:.4f}  ({sensitivity:.1%})  [Recall / TPR]\n")
    f.write(f"Specificity     : {specificity:.4f}  ({specificity:.1%})  [TNR]\n")
    f.write(f"Precision (PPV) : {precision:.4f}  ({precision:.1%})\n")
    f.write(f"NPV             : {npv:.4f}  ({npv:.1%})\n\n")
    f.write("CONFUSION MATRIX  (at threshold " + f"{threshold:.4f})\n" + "-"*40 + "\n")
    f.write(f"               Pred Normal    Pred TB\n")
    f.write(f"  True Normal     {TN:6d}     {FP:6d}   (FPR = {FP/(TN+FP):.1%})\n")
    f.write(f"  True TB         {FN:6d}     {TP:6d}   (TPR = {TP/(FN+TP):.1%})\n\n")
    f.write("CLASSIFICATION REPORT\n" + "-"*40 + "\n")
    f.write(classification_report(all_labels, all_preds,
                                  target_names=["Normal", "TB"], digits=4))
    f.write("\n\nFILES SAVED\n" + "-"*40 + "\n")
    f.write(f"Plot    : {out_png}\n")
    f.write(f"Summary : {summary_path}\n")

print(f"  Saved -> {summary_path}")
print(f"\n{'='*58}")
print(f"  Evaluation complete!")
print(f"  Open: {out_png}")
print(f"{'='*58}\n")