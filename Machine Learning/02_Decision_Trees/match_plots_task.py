import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles

# Generate sample dataset
np.random.seed(42)
X, y = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=42)

# Define different tree configurations
configurations = [
    ("Decision Tree, max_depth=1", 1),
    ("Decision Tree, max_depth=2", 2),
    ("Decision Tree, max_depth=3", 3),
    ("Decision Tree, max_depth=5", 5),
    ("Decision Tree, max_depth=10", 10)
]

# Create mesh for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (title, depth) in enumerate(configurations):
    # Train Decision Tree classifier
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
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
**Problem: Decision Tree Classification Boundaries (10 points)**

The following five plots show Decision Tree classifiers with different maximum depths. Your task is to match each plot (Model 1 to Model 5) to the corresponding configuration below.

a) Decision Tree, max_depth=1
b) Decision Tree, max_depth=2
c) Decision Tree, max_depth=3
d) Decision Tree, max_depth=5
e) Decision Tree, max_depth=10

✔️ Choose the correct model number (Model 1–5) for each configuration above.
"""

print(problem_text) 