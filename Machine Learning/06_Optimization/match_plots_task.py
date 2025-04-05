import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Define the gradient of Rosenbrock function
def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

# Define different optimization methods
methods = [
    ("Gradient Descent", "CG"),
    ("BFGS", "BFGS"),
    ("Newton-CG", "Newton-CG"),
    ("L-BFGS-B", "L-BFGS-B"),
    ("Nelder-Mead", "Nelder-Mead")
]

# Create contour plot of Rosenbrock function
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosenbrock([X[i, j], Y[i, j]])

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

# Starting point
x0 = np.array([-1.5, 2.0])

for i, (title, method) in enumerate(methods):
    # Track optimization path
    path = [x0]
    
    def callback(xk):
        path.append(xk)
    
    # Run optimization
    result = minimize(rosenbrock, x0, method=method, jac=rosenbrock_grad,
                     callback=callback, options={'maxiter': 100})
    
    # Plot contour and path
    axs[i].contour(X, Y, Z, levels=20, cmap='viridis')
    path = np.array(path)
    axs[i].plot(path[:, 0], path[:, 1], 'r-', linewidth=2)
    axs[i].scatter(path[:, 0], path[:, 1], c='r', s=20)
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("x1")
    axs[i].set_ylabel("x2")

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: Optimization Methods (10 points)**

The following five plots show different optimization methods applied to the Rosenbrock function. Your task is to match each plot (Model 1 to Model 5) to the corresponding optimization method below.

a) Gradient Descent (Conjugate Gradient)
b) BFGS
c) Newton-CG
d) L-BFGS-B
e) Nelder-Mead

✔️ Choose the correct model number (Model 1–5) for each optimization method above.
"""

print(problem_text) 