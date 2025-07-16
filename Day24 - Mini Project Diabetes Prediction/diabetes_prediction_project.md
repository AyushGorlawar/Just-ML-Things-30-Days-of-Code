# Diabetes Prediction – Mini Project

## Objective:
To predict whether a patient has diabetes based on diagnostic measurements using ML algorithms.

---

## Steps:
1. Loaded the dataset and handled missing values (zeros in Glucose, BMI, etc.)
2. Performed scaling using `StandardScaler`
3. Trained 3 models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest
4. Evaluated using:
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curve

---

## Observations:
- Logistic Regression performed well but KNN was slightly unstable for small k.
- Random Forest gave the highest ROC-AUC.
- Scaling was crucial for KNN performance.

---

## Conclusion:
Random Forest is the best-performing model for this dataset based on accuracy and recall.
