import numpy as np
import pandas as pd

WINDOW_SIZE = 30   # frames (~3 sec)
STRIDE = 15

EAR_TH = 0.25
MAR_TH = 0.6

def label_sequence(seq):
    ear_mean = seq[:, 0].mean()
    mar_max = seq[:, 1].max()
    nods = seq[:, 2].max()

    if ear_mean < EAR_TH:
        return 1   # DROWSY
    elif mar_max > MAR_TH or nods > 0:
        return 2   # FATIGUE
    else:
        return 0   # ALERT

def build_sequences(csv_path):
    df = pd.read_csv(csv_path)
    data = df[["ear", "mar", "nod"]].values

    X, y = [], []

    for i in range(0, len(data) - WINDOW_SIZE, STRIDE):
        seq = data[i:i+WINDOW_SIZE]
        label = label_sequence(seq)

        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)


X, y = build_sequences("data/features/features.csv")

np.save("data/sequences/X.npy", X)
np.save("data/sequences/y.npy", y)

print("Saved sequences:", X.shape, y.shape)
