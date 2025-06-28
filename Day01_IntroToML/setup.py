# Day 1 – ML Setup & Environment Test

# ✅ Check Python version
import sys
print("✅ Python version:", sys.version)

# ✅ Check key ML libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ Matplotlib version:", plt.__version__)
print("✅ Seaborn version:", sns.__version__)
print("✅ Scikit-learn version:", sklearn.__version__)

# ✅ Create sample data and simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="y = sin(x)", color="blue")
plt.title("Test Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Simple ML model: Linear Regression
from sklearn.linear_model import LinearRegression

# Fake data
X = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9])  # y = 2x + 1

model = LinearRegression()
model.fit(X, y)

print("\n✅ Linear Regression Test:")
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for x=5:", model.predict([[5]])[0])
