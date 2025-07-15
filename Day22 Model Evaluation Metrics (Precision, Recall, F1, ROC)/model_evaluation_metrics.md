# Model Evaluation Metrics

---

## ⚠ Why not just Accuracy?

Accuracy = (TP + TN) / Total

But in imbalanced data (e.g., fraud detection), accuracy can be misleading.

---

##  Better Metrics:

- **Precision** → How many predicted positives were correct?  
  `Precision = TP / (TP + FP)`

- **Recall** → How many actual positives were captured?  
  `Recall = TP / (TP + FN)`

- **F1 Score** → Harmonic mean of precision and recall  
  `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

- **ROC-AUC** → Measures model’s ability to distinguish classes  
  - Plot TPR vs FPR
  - AUC close to 1 is best

---

## Tip:
Use **F1** when **false negatives** and **false positives** are both important.
Use **Recall** when missing positives is costly (like cancer or fraud).

