import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# --- Correlation Matrix ---
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr[['target']].sort_values(by='target', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation of Features with Target")
plt.show()

# --- Chi-Square Test (on non-negative features) ---
X = df[data.feature_names]
y = df['target']
chi_selector = SelectKBest(score_func=chi2, k=5)
X_kbest = chi_selector.fit_transform(X, y)

# Show top features
chi_scores = pd.DataFrame({'Feature': data.feature_names, 'Chi2 Score': chi_selector.scores_})
print("\nTop Features by Chi-Square Score:\n", chi_scores.sort_values(by='Chi2 Score', ascending=False).head())
