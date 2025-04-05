import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

# Generate sample dataset
np.random.seed(42)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=42)

# Define different clustering algorithms
configurations = [
    ("K-means (k=4)", KMeans(n_clusters=4, random_state=42)),
    ("DBSCAN (eps=0.5, min_samples=5)", DBSCAN(eps=0.5, min_samples=5)),
    ("Agglomerative Clustering (k=4)", AgglomerativeClustering(n_clusters=4)),
    ("Gaussian Mixture (k=4)", GaussianMixture(n_components=4, random_state=42)),
    ("Spectral Clustering (k=4)", SpectralClustering(n_clusters=4, random_state=42))
]

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (title, clusterer) in enumerate(configurations):
    # Fit clustering algorithm
    if isinstance(clusterer, GaussianMixture):
        labels = clusterer.fit_predict(X)
    else:
        labels = clusterer.fit_predict(X)
    
    # Plot clusters
    scatter = axs[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("Feature 1")
    axs[i].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: Clustering Algorithms (10 points)**

The following five plots show different clustering algorithms applied to the same dataset. Your task is to match each plot (Model 1 to Model 5) to the corresponding clustering algorithm below.

a) K-means with k=4
b) DBSCAN with eps=0.5 and min_samples=5
c) Agglomerative Clustering with k=4
d) Gaussian Mixture Model with k=4
e) Spectral Clustering with k=4

✔️ Choose the correct model number (Model 1–5) for each clustering algorithm above.
"""

print(problem_text) 