
# Day 25 – Model Evaluation Metrics

This project focuses on understanding and applying various evaluation metrics to assess the performance of classification models. We utilize the Breast Cancer dataset available from `sklearn.datasets` and implement a Logistic Regression model for classification.

## Metrics Covered

The following metrics are used to evaluate the model's performance:

* **Accuracy**: Measures the overall correctness of the model.
* **Precision**: Indicates the proportion of positive identifications that were actually correct.
* **Recall**: Reflects the ability of the model to identify all relevant instances.
* **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced datasets.
* **Confusion Matrix**: Displays the performance of the classification model in a tabular format.
* **Classification Report**: Provides a detailed breakdown of precision, recall, F1-score, and support for each class.

## Dataset Used

* **Breast Cancer Dataset** from the `sklearn.datasets` module.
* Features include tumor measurements and characteristics used to predict whether the tumor is malignant or benign.

## Model Used

* **Logistic Regression** from `sklearn.linear_model`.

## Objective

The goal is to demonstrate how different evaluation metrics can provide deeper insights into model performance, especially in scenarios where accuracy alone may be misleading.


## Output Summary (from execution):
- Accuracy: 95.6%

- Precision: 94.5%

- Recall: 98.6%

- F1 Score: 96.5%
