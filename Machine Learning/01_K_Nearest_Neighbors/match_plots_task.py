import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

# Generate sample dataset
np.random.seed(42)
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# Define different distance metrics and k values
configurations = [
    ("Euclidean distance, k=1", "euclidean", 1),
    ("Manhattan distance, k=1", "manhattan", 1),
    ("Euclidean distance, k=5", "euclidean", 5),
    ("Manhattan distance, k=5", "manhattan", 5),
    ("Euclidean distance, k=15", "euclidean", 15)
]

# Create mesh for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (title, metric, k) in enumerate(configurations):
    # Train KNN classifier
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    clf.fit(X, y)
    
    # Predict on mesh points
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and training points
    axs[i].contourf(xx, yy, Z, alpha=0.4)
    axs[i].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("Feature 1")
    axs[i].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: K-Nearest Neighbors Classification Boundaries (10 points)**

The following five plots show K-Nearest Neighbors classifiers with different configurations. Your task is to match each plot (Model 1 to Model 5) to the corresponding configuration below.

a) Euclidean distance, k=1
b) Manhattan distance, k=1
c) Euclidean distance, k=5
d) Manhattan distance, k=5
e) Euclidean distance, k=15

✔️ Choose the correct model number (Model 1–5) for each configuration above.
"""

print(problem_text) 