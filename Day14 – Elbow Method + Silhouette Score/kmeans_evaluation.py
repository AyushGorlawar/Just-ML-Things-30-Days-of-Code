import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Elbow method
inertias = []
silhouette_scores = []

k_range = range(2, 11)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    model.fit(X)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, model.labels_))

# Plot elbow
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'o-')
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("Inertia (WCSS)")

# Plot silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-', color='green')
plt.title("Silhouette Score")
plt.xlabel("No. of Clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()
