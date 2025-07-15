# Scikit-learn ML Pipeline – Theory

---

## What is a Pipeline?

A `Pipeline` chains together multiple steps like:
1. Preprocessing (scaling, encoding)
2. Model training (e.g., logistic regression, random forest)

---

## 💡 Why Pipelines?

- Cleaner code
- Less chance of data leakage
- Reproducible and maintainable
- Works seamlessly with `GridSearchCV`

---

##  Typical Steps:
```python
Pipeline([
  ('scaler', StandardScaler()),
  ('classifier', LogisticRegression())
])
```
