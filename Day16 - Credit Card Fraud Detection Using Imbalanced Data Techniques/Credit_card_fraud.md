# Day 16 – Credit Card Fraud Detection Using Imbalanced Data Techniques

##  Project Overview
Detecting fraudulent credit card transactions is a highly critical real-world problem, especially due to the **imbalance in datasets** — where fraud cases make up less than 0.2% of all records.

This project demonstrates how to:
- Handle imbalanced data using **SMOTE**
- Train and compare models like **Logistic Regression, Random Forest**, and **XGBoost**
- Evaluate models using **Recall**, **Precision**, **F1-Score**, and **ROC-AUC**

##  Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- Features: `V1` to `V28` (PCA transformed), `Time`, `Amount`
- Target: `Class` → 0 = Legit, 1 = Fraud

##  Libraries Used
```python
pandas, numpy, matplotlib, seaborn
sklearn (LogisticRegression, RandomForestClassifier, metrics)
xgboost
imblearn.over_sampling.SMOTE
```

##  Project Workflow
1. **Load and Explore the Dataset**
2. **Handle Imbalance with SMOTE**
3. **Feature Scaling (`Amount` and `Time`)**
4. **Train/Test Split with Stratification**
5. **Train 3 Models**: Logistic Regression, Random Forest, XGBoost
6. **Model Evaluation** using:
   - Classification Report
   - ROC AUC Score
   - ROC Curve

##  Evaluation Metrics
We used the following to assess performance:
- **Recall**: Catching actual frauds
- **Precision**: Avoiding false alarms
- **F1-Score**: Balance of both
- **ROC-AUC**: Overall capability of classifier

##  Key Insights
- Logistic Regression is fast but not always ideal for imbalanced problems.
- Random Forest improves recall, handles imbalance better.
- XGBoost performs best with ROC AUC ~0.98 after SMOTE.

##  ROC Curve Example (XGBoost)
_After applying SMOTE + training XGBoost:_
```python
plt.plot(fpr, tpr, label='XGBoost')
plt.plot([0,1], [0,1], 'k--')
```

##  Folder Structure
```
Day16_CreditCard_Fraud_Detection/
├── fraud_detection.ipynb
├── dataset.csv (or Kaggle link)
└── README.md
```

##  Ethical Note
Use this project only for **educational & research purposes**. Never deploy fraud detection systems without real-time, regulated datasets and privacy safeguards.

##  Author
Made by [Ayush Gorlawar](https://github.com/AyushGorlawar)  
 
