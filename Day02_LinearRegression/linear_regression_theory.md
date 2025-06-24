# 📘 Linear Regression – Theory

Linear Regression is a **supervised learning algorithm** used to predict a continuous output based on input features.

---

## 🧮 Equation

> y = mx + c

- `y`: predicted value
- `x`: input feature
- `m`: slope (coefficient)
- `c`: intercept

In multiple dimensions:  
> y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

---

## 🎯 Objective

To find the best-fitting line by minimizing the **Mean Squared Error (MSE)**:
> MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

---

## 🧠 Use Cases

- Predicting house prices
- Estimating salaries
- Forecasting sales

---

## 📊 Evaluation Metric

- **R² Score** (Coefficient of Determination)
  - Ranges from 0 to 1
  - Closer to 1 = better model fit

---

📌 For code implementation, see the notebook `linear_regression_sklearn.ipynb`.
