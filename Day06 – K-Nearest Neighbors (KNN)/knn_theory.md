# 📘 K-Nearest Neighbors (KNN) – Theory

KNN is a **lazy, instance-based supervised learning algorithm** used for classification and regression.

---

## 💡 How It Works

1. Choose `k` (number of neighbors)
2. Calculate distance (e.g., Euclidean)
3. Find k-nearest points
4. Take majority class (for classification)

---

## 📊 Use Cases

- Recommender Systems
- Handwriting Recognition (like MNIST)
- Anomaly Detection

---

## ⚠️ Things to Note

- Sensitive to feature scale → use `StandardScaler`
- Best `k` is often chosen via cross-validation
