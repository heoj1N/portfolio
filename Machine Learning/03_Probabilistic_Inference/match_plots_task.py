import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Define different GMM configurations
configurations = [
    ("Single Gaussian, σ=0.5", 1, [0.5]),
    ("Single Gaussian, σ=1.0", 1, [1.0]),
    ("Two Gaussians, equal weights", 2, [0.5, 0.5]),
    ("Two Gaussians, unequal weights", 2, [0.3, 0.7]),
    ("Three Gaussians", 3, [0.3, 0.3, 0.4])
]

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
x = np.linspace(-5, 5, 1000)

for i, (title, n_components, weights) in enumerate(configurations):
    if n_components == 1:
        # Single Gaussian case
        y = norm.pdf(x, 0, weights[0])
        axs[i].plot(x, y, 'b-', linewidth=2)
    else:
        # Multiple Gaussians case
        means = np.linspace(-2, 2, n_components)
        y = np.zeros_like(x)
        for j in range(n_components):
            y += weights[j] * norm.pdf(x, means[j], 0.5)
        axs[i].plot(x, y, 'b-', linewidth=2)
    
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("Probability Density")
    axs[i].set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: Gaussian Mixture Models (10 points)**

The following five plots show different probability density functions. Your task is to match each plot (Model 1 to Model 5) to the corresponding configuration below.

a) Single Gaussian with standard deviation σ=0.5
b) Single Gaussian with standard deviation σ=1.0
c) Two Gaussians with equal weights
d) Two Gaussians with unequal weights (0.3 and 0.7)
e) Three Gaussians with weights (0.3, 0.3, 0.4)

✔️ Choose the correct model number (Model 1–5) for each configuration above.
"""

print(problem_text) 