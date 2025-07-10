# Day 16 – Credit Card Fraud Detection Using Imbalanced Data Techniques

# 📦 Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE

# 📥 Step 2: Load Dataset
data = pd.read_csv('creditcard.csv')

# 📊 Step 3: Exploratory Data Analysis
print("Class Distribution:\n", data['Class'].value_counts())
plt.figure(figsize=(5,4))
sns.countplot(data['Class'])
plt.title("Class Distribution")
plt.show()

# 📈 Step 4: Feature Scaling
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time','Amount'], axis=1, inplace=True)
data = data[['scaled_time','scaled_amount'] + [col for col in data.columns if col not in ['scaled_time','scaled_amount','Class']] + ['Class']]

# 🎯 Step 5: Split Data
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ⚠️ Step 6: Handle Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 🧠 Step 7: Train Models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

lr.fit(X_train_sm, y_train_sm)
rf.fit(X_train_sm, y_train_sm)
xgb.fit(X_train_sm, y_train_sm)

# 📊 Step 8: Evaluate Models
models = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}

for name, model in models.items():
    print(f"\n🔍 {name} Results:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# 📈 Step 9: ROC Curve for Best Model (XGBoost)
y_probs = xgb.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr, label='XGBoost')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.show()
