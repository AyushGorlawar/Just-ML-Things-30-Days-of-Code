import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Generate data
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2, random_state=42)

# Plot dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.grid()
plt.show()

# Fit Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_pred = model.fit_predict(X)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred, palette="Set2", s=60)
plt.title("Hierarchical Clustering (Agglomerative)")
plt.grid()
plt.tight_layout()
plt.show()
