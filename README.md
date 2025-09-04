# Sign Language MNIST Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify American Sign Language (ASL) letters using the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).  
Achieved **~99.7% validation accuracy** in under 10 epochs.

---

## 📖 Project Overview
The Sign Language MNIST dataset contains **28×28 grayscale images** of hand signs representing **24 ASL letters** (letters `J` and `Z` are excluded because they require motion).  
This project:
- Preprocesses the dataset (normalization, reshaping, label remapping).
- Trains a CNN with data augmentation and learning rate scheduling.
- Evaluates model performance (accuracy, confusion matrix).
- Exports the model for deployment (Keras + TFLite).

---

## 🗂️ Dataset
- **Train samples:** 27,455  
- **Test samples:** 7,172  
- **Classes:** 24 (A–Y, excluding J and Z)  
- **Format:** CSV (each row = label + 784 pixels)

---

## 🧠 Model Architecture
```text
Input (28×28×1)
→ Conv2D(32, 3×3) + ReLU + BatchNorm + MaxPool(2×2)
→ Conv2D(64, 3×3) + ReLU + BatchNorm + MaxPool(2×2)
→ Conv2D(128, 3×3) + ReLU + BatchNorm + MaxPool(2×2)
→ Flatten
→ Dense(256) + ReLU + Dropout(0.5)
→ Dense(24) + Softmax
