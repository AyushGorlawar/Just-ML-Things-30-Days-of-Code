import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create sample data
data = {
    'Age': [22, 25, np.nan, 29, 31, 120, 28, 30, np.nan, 27],
    'Salary': [25000, 27000, 30000, np.nan, 32000, 35000, 40000, np.nan, 28000, 29000]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)

# Handling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Handling outliers (IQR Method)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Age'] < Q1 - 1.5 * IQR) | (df['Age'] > Q3 + 1.5 * IQR)]
print("\nOutliers Detected in Age:\n", outliers)

# Remove outliers
df = df[~((df['Age'] < Q1 - 1.5 * IQR) | (df['Age'] > Q3 + 1.5 * IQR))]

# Plot
sns.boxplot(data=df, x='Age')
plt.title("Boxplot After Outlier Removal")
plt.show()
