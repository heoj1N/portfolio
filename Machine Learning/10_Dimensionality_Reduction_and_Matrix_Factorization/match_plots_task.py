import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP

# Generate sample dataset
np.random.seed(42)
X, color = make_swiss_roll(n_samples=1000, noise=0.2)

# Define different dimensionality reduction techniques
configurations = [
    ("PCA (2 components)", PCA(n_components=2)),
    ("Kernel PCA (RBF)", KernelPCA(n_components=2, kernel='rbf')),
    ("t-SNE", TSNE(n_components=2)),
    ("MDS", MDS(n_components=2)),
    ("UMAP", UMAP(n_components=2))
]

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (title, reducer) in enumerate(configurations):
    # Fit and transform data
    X_reduced = reducer.fit_transform(X)
    
    # Plot reduced data
    scatter = axs[i].scatter(X_reduced[:, 0], X_reduced[:, 1],
                            c=color, cmap='viridis')
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("Component 1")
    axs[i].set_ylabel("Component 2")

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: Dimensionality Reduction Techniques (10 points)**

The following five plots show different dimensionality reduction techniques applied to the Swiss Roll dataset. Your task is to match each plot (Model 1 to Model 5) to the corresponding technique below.

a) Principal Component Analysis (PCA)
b) Kernel PCA with RBF kernel
c) t-SNE
d) Multidimensional Scaling (MDS)
e) UMAP

✔️ Choose the correct model number (Model 1–5) for each dimensionality reduction technique above.
"""

print(problem_text) 