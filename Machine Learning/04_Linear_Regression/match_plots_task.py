import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

# Generate sample dataset
np.random.seed(0)
X = np.sort(np.random.rand(10, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Define models
models = [
    ("Polynomial regression (degree = 5), no regularization",
     make_pipeline(PolynomialFeatures(5), LinearRegression())),
    
    ("Polynomial regression (degree = 10), no regularization",
     make_pipeline(PolynomialFeatures(10), LinearRegression())),
    
    ("Polynomial regression (degree = 50), L2 regularization with λ = 10^3",
     make_pipeline(PolynomialFeatures(50), Ridge(alpha=1000))),
    
    ("Feed-forward neural network with ReLU activation, no regularization",
     MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=10000, random_state=1)),
    
    ("Decision tree of depth 2",
     DecisionTreeRegressor(max_depth=2, random_state=0))
]

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
x_plot = np.linspace(0, 1, 100).reshape(-1, 1)

for i, (desc, model) in enumerate(models):
    model.fit(X, y)
    y_pred = model.predict(x_plot)
    axs[i].scatter(X, y, facecolor="none", edgecolor="k")
    axs[i].plot(x_plot, y_pred, label="Prediction")
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: Regression Model Matching (10 points)**

The following five plots show five different regression models fitted to the same dataset. Your task is to assign each of the plots (Model 1 to Model 5) to the corresponding model below.

a) Polynomial regression (degree = 5), no regularization  
b) Polynomial regression (degree = 10), no regularization  
c) Polynomial regression (degree = 50), L2 regularization with λ = 10³  
d) Feed-forward neural network with ReLU activation functions, no regularization  
e) Decision tree of depth 2  

✔️ Choose the correct model number (Model 1–5) for each method above.
"""

print(problem_text)
