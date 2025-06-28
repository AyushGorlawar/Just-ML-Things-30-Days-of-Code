 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 🎯 Sample non-linear dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([3, 6, 11, 18, 27, 38])  

# 🔁 Generate polynomial features (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 🧠 Train the regression model
model = LinearRegression()
model.fit(X_poly, y)

# 🎯 Make predictions
y_pred = model.predict(X_poly)

# 📊 Evaluation
print("✅ Coefficients:", model.coef_)
print("✅ Intercept:", model.intercept_)
print("✅ R² Score:", r2_score(y, y_pred))

# 📈 Plotting actual vs predicted
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Polynomial Fit (Degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression Example")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
