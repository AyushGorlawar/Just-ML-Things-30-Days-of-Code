import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Sample data
df = pd.DataFrame({
    'Age': [22, 25, 28, 30],
    'Salary': [25000, 27000, 40000, 31000],
    'Gender': ['Male', 'Female', 'Female', 'Male'],
    'City': ['Mumbai', 'Delhi', 'Delhi', 'Mumbai']
})

# Standard Scaling
scaler = StandardScaler()
df[['Age_scaled', 'Salary_scaled']] = scaler.fit_transform(df[['Age', 'Salary']])

# Label Encoding
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['City'])

print(df_encoded)
