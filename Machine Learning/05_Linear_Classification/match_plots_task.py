import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Generate sample dataset
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Define different linear classifiers
configurations = [
    ("Logistic Regression, C=0.1", LogisticRegression(C=0.1)),
    ("Logistic Regression, C=1.0", LogisticRegression(C=1.0)),
    ("Logistic Regression, C=10.0", LogisticRegression(C=10.0)),
    ("Perceptron", Perceptron()),
    ("Linear SVM, C=1.0", SVC(kernel='linear', C=1.0))
]

# Create mesh for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (title, clf) in enumerate(configurations):
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
**Problem: Linear Classification Boundaries (10 points)**

The following five plots show different linear classifiers fitted to the same dataset. Your task is to match each plot (Model 1 to Model 5) to the corresponding classifier below.

a) Logistic Regression with regularization strength C=0.1
b) Logistic Regression with regularization strength C=1.0
c) Logistic Regression with regularization strength C=10.0
d) Perceptron
e) Linear SVM with regularization strength C=1.0

✔️ Choose the correct model number (Model 1–5) for each classifier above.
"""

print(problem_text) 