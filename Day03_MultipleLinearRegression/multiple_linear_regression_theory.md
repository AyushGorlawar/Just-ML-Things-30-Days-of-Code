# 📘 Multiple Linear Regression – Theory

Multiple Linear Regression is a **supervised learning algorithm** used to model the relationship between **one dependent variable (Y)** and **two or more independent variables (X₁, X₂, ..., Xₙ)**.

---

## 🧮 Equation

> y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ + ε

Where:
- `y` = predicted value
- `x₁, x₂, ..., xₙ` = independent variables
- `b₀` = intercept
- `b₁, b₂, ..., bₙ` = coefficients
- `ε` = error term

---

## 🧠 Why Use Multiple Linear Regression?

Because real-world outcomes often depend on **multiple factors**. Example:
- House Price = f(size, location, number of rooms)
- Salary = f(education level, experience, test score)

---

## 🎯 Goal

To estimate the coefficients (b₀ to bₙ) such that the line fits the data and minimizes the **Mean Squared Error (MSE)**.

---

## 📊 Evaluation Metric

- R² Score (Coefficient of Determination)
  - Indicates how well the model explains the variability in output

---

📌 For implementation, check `multiple_linear_regression_sklearn.ipynb`
