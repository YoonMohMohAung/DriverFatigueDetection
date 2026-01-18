import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import CNN_LSTM

# ======================
# Load data
# ======================
X = np.load("data/sequences/X.npy")
y = np.load("data/sequences/y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=32
)

# ======================
# Model
# ======================
model = CNN_LSTM(
    input_dim=X.shape[2],    # EAR, MAR, nod
    hidden_dim=64,
    num_classes=3
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# Training loop
# ======================
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            outputs = model(xb)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.3f} | Val Acc: {acc:.2f}")

# Save model
torch.save(model.state_dict(), "models/fatigue_cnn_lstm.pth")
print("Model saved.")
