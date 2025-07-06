import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize features
X_std = StandardScaler().fit_transform(X)

# Apply PCA (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']
for i, color in zip(range(3), colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.7, color=color, label=target_names[i])

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA on Iris Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
