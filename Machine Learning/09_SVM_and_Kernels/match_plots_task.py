import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# Generate sample dataset
np.random.seed(42)
X, y = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=42)

# Define different kernel configurations
configurations = [
    ("Linear kernel, C=1.0", "linear", 1.0),
    ("RBF kernel, C=1.0, γ=0.1", "rbf", 1.0, 0.1),
    ("RBF kernel, C=1.0, γ=1.0", "rbf", 1.0, 1.0),
    ("Polynomial kernel, C=1.0, degree=2", "poly", 1.0, 2),
    ("Polynomial kernel, C=1.0, degree=3", "poly", 1.0, 3)
]

# Create mesh for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, config in enumerate(configurations):
    if len(config) == 3:
        title, kernel, C = config
        clf = SVC(kernel=kernel, C=C)
    else:
        title, kernel, C, param = config
        if kernel == "rbf":
            clf = SVC(kernel=kernel, C=C, gamma=param)
        else:  # polynomial
            clf = SVC(kernel=kernel, C=C, degree=param)
    
    # Train classifier
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
**Problem: SVM Kernel Functions (10 points)**

The following five plots show SVM classifiers with different kernel functions. Your task is to match each plot (Model 1 to Model 5) to the corresponding kernel configuration below.

a) Linear kernel with C=1.0
b) RBF kernel with C=1.0 and γ=0.1
c) RBF kernel with C=1.0 and γ=1.0
d) Polynomial kernel with C=1.0 and degree=2
e) Polynomial kernel with C=1.0 and degree=3

✔️ Choose the correct model number (Model 1–5) for each kernel configuration above.
"""

print(problem_text) 