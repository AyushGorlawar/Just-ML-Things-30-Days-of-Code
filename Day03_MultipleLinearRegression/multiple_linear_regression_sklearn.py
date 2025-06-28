# Multiple Linear Regression with scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample dataset: Study hours, Practice tests ➝ Final score
data = {
    "Hours_Studied": [2, 4, 6, 8, 10],
    "Practice_Tests": [1, 2, 2, 3, 4],
    "Score": [50, 60, 65, 75, 85]
}

df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[["Hours_Studied", "Practice_Tests"]]
y = df["Score"]

# Model training
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y, y_pred))

# Prediction for new input
new_input = np.array([[9, 3]])  # 9 study hours, 3 practice tests
print("Prediction for [9 hrs, 3 tests]:", model.predict(new_input)[0])

# Visualization (2D approximation)
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted (Multiple Linear Regression)")
plt.grid(True)
plt.show()
