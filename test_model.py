import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

from tb_system.core.model import TBModel
from tb_system.core.preprocessing import TBDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
model = TBModel(num_classes=2, pretrained=False, use_metadata=False)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load test dataset
test_ds = TBDataset("data", "test", size=384)
loader = DataLoader(test_ds, batch_size=16, shuffle=False)

preds, targets, probs = [], [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits, _ = model(imgs, None)
        probabilities = torch.softmax(logits, dim=1)

        preds += logits.argmax(1).cpu().tolist()
        targets += labels.cpu().tolist()
        probs += probabilities[:, 1].cpu().tolist()

print("\n===== FINAL TEST RESULTS =====\n")
print(classification_report(targets, preds, target_names=["Normal", "TB"]))

auc = roc_auc_score(targets, probs)
print(f"ROC-AUC Score: {auc:.4f}")