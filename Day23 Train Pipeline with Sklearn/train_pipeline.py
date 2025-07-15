import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation scores
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("Cross-validation Accuracy:", scores.mean())

# Fit the pipeline and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nSample Prediction on Test Set:", y_pred[:5])
