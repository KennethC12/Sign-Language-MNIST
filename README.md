# ✋ Sign Language MNIST Classifier

A deep learning project that classifies American Sign Language (ASL) hand signs using a **Convolutional Neural Network (CNN)**.  
Trained on the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist), this model reaches **99.7% validation accuracy**.

---

## 🌟 Highlights
- 📚 Built and trained a CNN from scratch using **TensorFlow/Keras**  
- 🖼️ Worked with **28×28 grayscale images** of ASL letters  
- 🎯 Achieved **state-of-the-art level accuracy** (~99.7%)  
- 🔄 Applied **data augmentation, batch normalization, dropout** to prevent overfitting  
- 📱 Exported the model to **TensorFlow Lite (TFLite)** for mobile deployment  

---

## 📂 Dataset Overview
- **Source:** [Kaggle - Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)  
- **Train set:** 27,455 images  
- **Test set:** 7,172 images  
- **Classes:** 24 (letters A–Y, excluding J and Z since they require motion)  
- **Format:** CSV files (`label` + 784 pixel values)

---

## 🧠 Model Architecture

```text
Input (28×28×1 grayscale image)
│
├── Conv2D (32 filters, 3×3) + ReLU + BatchNorm + MaxPool(2×2)
├── Conv2D (64 filters, 3×3) + ReLU + BatchNorm + MaxPool(2×2)
├── Conv2D (128 filters, 3×3) + ReLU + BatchNorm + MaxPool(2×2)
│
├── Flatten
├── Dense (256) + ReLU + Dropout (0.5)
└── Dense (24) + Softmax
