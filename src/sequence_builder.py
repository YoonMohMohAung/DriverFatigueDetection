import os
import numpy as np
import pandas as pd

FEATURE_DIR = "data/features"
OUTPUT_DIR = "data/sequences"

SEQUENCE_LENGTH = 30   # frames per sequence (≈1 sec at 30fps)
STRIDE = 5             # overlap between sequences

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_label_from_filename(filename):
    filename = filename.lower()

    if filename.startswith("alert"):
        return 0
    elif filename.startswith("drowsy"):
        return 1
    elif filename.startswith("yawning"):
        return 2
    else:
        raise ValueError(f"Unknown label in filename: {filename}")

def build_sequences_from_csv(csv_path, label):
    df = pd.read_csv(csv_path)

    features = df[["ear", "mar", "nod"]].values
    sequences = []
    labels = []

    for start in range(0, len(features) - SEQUENCE_LENGTH + 1, STRIDE):
        end = start + SEQUENCE_LENGTH
        seq = features[start:end]

        sequences.append(seq)
        labels.append(label)

    return sequences, labels

def main():
    X, y = [], []

    for file in os.listdir(FEATURE_DIR):
        if not file.endswith(".csv"):
            continue

        label = get_label_from_filename(file)
        csv_path = os.path.join(FEATURE_DIR, file)

        seqs, labels = build_sequences_from_csv(csv_path, label)
        X.extend(seqs)
        y.extend(labels)

        print(f"[OK] {file}: {len(seqs)} sequences")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("\n=== DATASET SUMMARY ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()
