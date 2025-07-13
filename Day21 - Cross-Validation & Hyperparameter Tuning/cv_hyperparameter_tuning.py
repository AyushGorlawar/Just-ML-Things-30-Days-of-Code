import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with CV
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation Accuracy (5-fold):", scores.mean())

# Grid Search for tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 4]
}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n", classification_report(y_test, grid.predict(X_test)))
