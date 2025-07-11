# ⚙️ Data Scaling, Normalization & Encoding

---

## 📏 Scaling vs Normalization:

- **StandardScaler** → Zero mean, unit variance (Z-score)
- **MinMaxScaler** → Scales data between 0 and 1

---

## 🧠 Why?

- Many ML models (SVM, KNN, Logistic Regression) are sensitive to scale.

---

## 🔤 Encoding:

- **Label Encoding** → Converts categories to 0, 1, 2...
- **One-Hot Encoding** → Converts into binary columns

---

## 💡 Tip:
Apply **scaling after** missing value treatment, but **before** model training.
