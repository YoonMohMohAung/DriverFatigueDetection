# 🚗 Driver Fatigue Detection (CNN + LSTM)

A real-time **Driver Fatigue Detection System** using **Computer Vision** and **Deep Learning**.
The system detects **Alert**, **Drowsy**, and **Yawning** states from video or webcam input using **facial landmarks, temporal features, and a CNN + LSTM model**.

This project is designed as a **student / research-level implementation** and follows a clean, modular ML pipeline.

---

## 📌 Features

* Real-time webcam fatigue detection
* Supports video file testing
* Facial landmark-based feature extraction
* Temporal modeling using LSTM
* CNN-assisted feature learning
* Multi-class classification:

  * `0 → Alert`
  * `1 → Drowsy`
  * `2 → Yawning`
* Model evaluation with accuracy, precision, recall, F1-score

---

## 🧠 System Overview

### Pipeline

```
Video / Webcam
   ↓
Face Detection & Landmarks (MediaPipe)
   ↓
Feature Extraction (EAR, MAR, Head Nod)
   ↓
Sliding Window (Temporal Sequences)
   ↓
CNN + LSTM Model
   ↓
Driver State Prediction
```

### Why CNN + LSTM?

* **CNN**: Learns spatial patterns from facial features
* **LSTM**: Captures temporal behavior (eye closure duration, yawning events)

---

## 📁 Project Structure

```
DriverFatigueDetection/
│
├── data/
│   ├── raw_videos/          # Original videos (alert / drowsy / yawning)
│   ├── features/            # Extracted per-frame features
│   └── sequences/           # X.npy, y.npy (temporal sequences)
│
├── models/
│   └── fatigue_cnn_lstm.pth # Trained model
│
├── src/
│   ├── extract_features.py  # Extract EAR, MAR, head nod
│   ├── build_sequences.py   # Build LSTM input sequences
│   ├── train.py             # Model training
│   ├── test_cnn_lstm.py     # Offline testing
│   ├── main_webcam.py       # Real-time webcam testing
│   ├── model.py             # CNN + LSTM architecture
│   ├── landmarks.py         # MediaPipe landmark detection
│   └── utils.py             # Feature calculations
│
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset

### Minimum Recommendation

* **15–20 short videos** (15–30 seconds each)
* Multiple people
* Different lighting conditions

### Class Labels

| Label | Meaning |
| ----- | ------- |
| 0     | Alert   |
| 1     | Drowsy  |
| 2     | Yawning |

⚠️ **Yawning is an event**, while **drowsiness is a state**. Ensure clean labeling.

---

## ⚙️ Installation

### 1️⃣ Create virtual environment (optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Required libraries

* Python 3.9+
* OpenCV
* MediaPipe
* PyTorch
* NumPy
* Scikit-learn

---

## 🚀 Usage

### Step 1: Extract features

```bash
python src/extract_features.py
```

### Step 2: Build sequences

```bash
python src/build_sequences.py
```

### Step 3: Train model

```bash
python src/train.py
```

### Step 4: Test model (offline)

```bash
python src/test_cnn_lstm.py
```

### Step 5: Real-time webcam test

```bash
python src/main_webcam.py
```

Press **Q** to quit.

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Due to dataset size, results may vary. More data = better performance.

---

## ⚠️ Known Limitations

* Small dataset leads to class confusion
* Yawning and drowsiness may overlap
* Sensitive to lighting and face angle

---

## 🔧 Future Improvements

* Larger and more diverse dataset
* Separate yawning event detector
* Fatigue score instead of hard labels
* Attention-based LSTM
* Mobile / embedded deployment

---

## 📚 References

* MediaPipe Face Mesh
* NTHU Drowsy Driver Dataset
* YawDD Dataset
* UTA-RLDD Dataset

---

## 👨‍💻 Author

**Driver Fatigue Detection Project**
Developed for learning, research, and demonstration purposes.

---

## ⭐ Acknowledgment

This project combines classical computer vision with deep learning to demonstrate a practical fatigue detection system.

---

> ⚠️ Disclaimer: This system is for educational and research use only. Not intended for real-world safety-critical deployment.
