import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import CNN_LSTM

# ======================
# Config
# ======================
MODEL_PATH = "models/fatigue_cnn_lstm.pth"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Load data
# ======================
X = np.load("data/sequences/X.npy")
y = np.load("data/sequences/y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# SAME split strategy as training
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ======================
# Load model
# ======================
model = CNN_LSTM(
    input_dim=X.shape[2],   # MUST match training
    hidden_dim=64,
    num_classes=3
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# Testing
# ======================
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        outputs = model(xb)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# ======================
# Metrics
# ======================
accuracy = accuracy_score(all_labels, all_preds)
print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
