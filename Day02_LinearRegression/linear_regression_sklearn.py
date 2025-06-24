# Linear Regression with sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # y = 2x

# Model Training
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Evaluation
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("R² Score:", r2_score(y, y_pred))

# Visualization
plt.scatter(X, y, color='blue', label="Actual")
plt.plot(X, y_pred, color='red', label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.grid(True)
plt.show()
