# 📘 Polynomial Regression – Theory

Polynomial Regression is a **supervised learning** algorithm used when the relationship between input and output is **non-linear** but can be approximated by a polynomial.

---

## 🧮 Equation

> y = b₀ + b₁x + b₂x² + b₃x³ + ... + bₙxⁿ + ε

Where:
- `y` = target value
- `x` = input feature
- `b₀ ... bₙ` = coefficients
- `n` = degree of the polynomial

---

## 🧠 When to Use?

- When data shows a curve-like trend, not a straight line
- Linear regression underfits the data

---

## ⚙️ How It Works?

- Use `PolynomialFeatures(degree=n)` from `sklearn.preprocessing` to generate additional features like x², x³
- Then use a normal `LinearRegression()` model on the transformed feature set

---

## 📊 Evaluation

- R² Score
- Visual inspection of curve fitting

---

📌 For code implementation, check `polynomial_regression_sklearn.py`
