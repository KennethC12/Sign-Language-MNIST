# ASL Sign Language (MNIST) — CNN Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/placeholder/ASL-CNN/blob/main/asl_sign_mnist_colab.ipynb)

A convolutional neural network (CNN) that classifies American Sign Language (ASL) hand signs using the **Sign Language MNIST** dataset (`sign_mnist_train.csv`, `sign_mnist_test.csv`).  
This repo includes:
- Clean, reproducible training script (`train_asl_cnn.py`)
- Data preprocessing (normalization, label compaction because **J** and **Z** are missing)
- Augmentation (`ImageDataGenerator`), LR scheduling, and early stopping
- Evaluation with classification report & confusion matrix
- Artifact saving: trained Keras model and label map JSON

> **Result**: With the provided architecture and augmentations, validation accuracy ~**99%** on the test split (as shown in the reference training logs).

---

## 📦 Dataset

Download from Kaggle or sources that provide **Sign Language MNIST** as two CSV files:
- `sign_mnist_train.csv`
- `sign_mnist_test.csv`

Each row has a `label` column (0–25 letters but **J** and **Z** are omitted in this dataset) and 784 pixel values for a 28×28 grayscale image.

Place both CSVs in the working directory (or pass a custom `--data_dir`).

---

## 🚀 Quickstart

### 1) Local (Python/CLI)
```bash
# (Optional) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train
python train_asl_cnn.py --data_dir . --epochs 20 --batch_size 64
```
Artifacts are written to `artifacts/`:
- `asl_cnn.keras` — trained model
- `label_map.json` — original→compact label map
- `accuracy.png`, `loss.png` — training curves
- `classification_report.txt`
- `confusion_matrix.png`

### 2) Google Colab
1. Click the **Open in Colab** badge above.
2. Upload `sign_mnist_train.csv` and `sign_mnist_test.csv` to Colab (or mount Drive).
3. Run all cells.

---

## 🧠 Model

```
Input (28×28×1)
→ Conv(32, 3×3) + BN + MaxPool(2)
→ Conv(64, 3×3) + BN + MaxPool(2)
→ Conv(128, 3×3) + BN + MaxPool(2)
→ Flatten
→ Dense(256, ReLU) + Dropout(0.5)
→ Dense(24, Softmax)
```

- Optimizer: **Adam**
- Loss: **Categorical Cross-Entropy**
- Metrics: **Accuracy**
- Augmentation: rotation/shift/zoom

---

## 📊 Results (example run)

```
Test accuracy: 0.9968 | Test loss: 0.0090
```

The `classification_report.txt` and `confusion_matrix.png` are generated under `artifacts/` after a run.

---

## 🔤 Labels & Mapping

The dataset excludes **J** and **Z**. We remap raw labels to a compact range `0..23`.  
A helper JSON (`label_map.json`) is saved for downstream use. We also produce an `id_to_letter` mapping for A..Z minus J/Z.

---

## 🧪 TTA Inference (optional)

The script includes `predict_tta(x, n=5)` for **test-time augmentation** averaging. You can call it with a batch of images (shape `(N, 28, 28, 1)`).

---

## ⚠️ Tips & Troubleshooting

- **OOM / Slow training**: Lower `--batch_size`, reduce augmentation, or use fewer epochs.
- **Colab GPU**: Ensure `Runtime → Change runtime type → GPU`.
- **CSV path errors**: Ensure `--data_dir` contains both CSVs with the exact names.
- **Reproducibility**: Set `--seed` to fix RNG seeds.

---

## 📄 License

MIT (or your preferred license).
