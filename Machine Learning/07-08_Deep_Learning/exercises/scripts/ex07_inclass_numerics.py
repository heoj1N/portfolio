import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import minmax_scale

from scipy.spatial.distance import cdist, pdist, squareform

# # In-class exercise 7: Deep Learning 1 (Part A) - Numerics

def sigmoid(t):
    """Apply sigmoid to the input array."""
    return 1 / (1 + np.exp(-t))

def predict_for_loop(X, w, b):
    """Generate predictions with a logistic regression model using a for-loop.

    Args:
        X: data matrix, shape (N, D)
        w: weights vector, shape (D)
        b: bias term, shape (1)

    Returns:
        y: probabilies of the positive class, shape (N)
    """
    n_samples = X.shape[0]
    y = np.zeros([n_samples])
    for i in range(n_samples):
        score = np.dot(X[i], w) + b
        y[i] = sigmoid(score)
    return y

def predict_vectorized(X, w, b):
    """Generate predictions with a logistic regression model using vectorized operations.

    Args:
        X: data matrix, shape (N, D)
        w: weights vector, shape (D)
        b: bias term, shape (1)

    Returns:
        y: probabilies of the positive class, shape (N)
    """
    scores = X @ w + b
    y = sigmoid(scores)
    return y

def l2_distance(x, y):
    """Compute Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x - y) ** 2))

def distances_for_loop(X):
    """Compute pairwise distances between all instances (for loop version).

    Args:
        X: data matrix, shape (N, D)

    Returns:
        dist: matrix of pairwise distances, shape (N, N)
    """
    n_samples = X.shape[0]
    distances = np.zeros([n_samples, n_samples])
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = l2_distance(X[i], X[j])
    return distances

def distances_vectorized(X):
    """Compute pairwise distances between all instances (vectorized version).

    Args:
        X: data matrix, shape (N, D)

    Returns:
        dist: matrix of pairwise distances, shape (N, N)
    """
    return np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=-1))

def softmax_unstable(logits):
    """Compute softmax values for each sets of scores in logits."""
    exp_scores = np.exp(logits)
    return exp_scores / np.sum(exp_scores, axis=0)

def softmax_stable(logits):
    """Compute softmax values for each sets of scores in logits."""
    logits_shifted = logits - np.max(logits, axis=0)
    denominator = np.sum(np.exp(logits_shifted), axis=0)
    return np.exp(logits_shifted) / denominator

def binary_cross_entropy_unstable(scores, labels):
    """Compute binary cross-entropy loss for one sample."""
    return -labels * np.log(sigmoid(scores)) - (1 - labels) * np.log(
        1 - sigmoid(scores)
    )

def binary_cross_entropy_stable(scores, labels):
    """Compute binary cross-entropy loss for one sample."""
    return np.log(1 + np.exp(scores)) - labels * scores

def log_sigmoid_unstable(x):
    """Compute log sigmoid for one sample."""
    return np.log(1 / (1 + np.exp(-x)))

def log_sigmoid_stable(x):
    """Compute log sigmoid for one sample."""
    return -np.log1p(np.exp(-x))

def main():

    X, y = load_breast_cancer(return_X_y=True)
    X = minmax_scale(X, feature_range=(-1, 1)) # Scale each feature to [-1, 1] range
    X.shape, y.shape

    n_features = X.shape[1]
    w = np.random.normal(size=[n_features], scale=0.1)  # weight vector
    b = np.random.normal(size=[1])  # bias

    print(sigmoid(0)) # input is a scalar
    print(sigmoid(np.array([0, 1, 2]))) # input is a vector
    print(sigmoid(np.array([[0, 1, 2], [-1, -2, -3]]))) # input is a matrix

    t = np.array([0, 1, 2]) # How does broadcasting work between a scalar and a vector?

    # Let's analyse the function 1 + np.exp(-t)
    # 1 is a scalar, so it is broadcasted to [1, 1, 1]. Let's see how
    a = np.array(1)

    print(a[np.newaxis].repeat(t.shape[0], axis=0))
    print(a[np.newaxis].repeat(t.shape[0], axis=0).reshape(t.shape[0]))
    print(a[np.newaxis].repeat(t.shape[0], axis=0).reshape(t.shape[0]) + t)

    # How does broadcasting work between a scalar and a matrix?
    t = np.array([[0, 1, 2], [-1, -2, -3]])
    a = np.array(1)
    print(a[np.newaxis])
    print(a[np.newaxis].repeat(t.shape[1], axis=0))
    print(a[np.newaxis].repeat(t.shape[1], axis=0)[np.newaxis])
    print(a[np.newaxis].repeat(t.shape[1], axis=0)[np.newaxis].repeat(t.shape[0], axis=0))
    print(
        a[np.newaxis].repeat(t.shape[1], axis=0)[np.newaxis].repeat(t.shape[0], axis=0) + t
    )

    # How does broadcasting work between a vector and a matrix?
    t = np.array([[0, 1, 2], [-1, -2, -3]])
    v = np.array([1, 2, 3])
    
    # add dimension to v
    print(v[np.newaxis, :])
    print(v[np.newaxis, :].repeat(t.shape[0], axis=0))
    print(v[np.newaxis, :].repeat(t.shape[0], axis=0) + t)

    results_for_loop = predict_for_loop(X, w, b)
    results_vectorized = predict_vectorized(X, w, b)

    np.all(results_for_loop == results_vectorized)
    np.linalg.norm(results_for_loop - results_vectorized)
    np.allclose(results_for_loop, results_vectorized)

    predict_for_loop(X, w, b)
    predict_vectorized(X, w, b)

    # compute pairwise distances using for loops
    dist1 = distances_for_loop(X)
    x = np.arange(5, dtype=np.float64)

    print(x[:, np.newaxis].repeat(x.shape[0], axis=1))
    print(x[np.newaxis, :].repeat(x.shape[0], axis=0))
    print(x[:, np.newaxis] - x[np.newaxis, :])
    print(x[:, None] - x[None, :])
    print(np.sum(np.square(x[:, None] - x[None, :]), -1))

    dist2 = distances_vectorized(X) # compute pairwise distances using vectorized operations
    np.allclose(dist1, dist2)

    dist3 = cdist(X, X)
    dist4 = squareform(pdist(X))
    np.allclose(dist2, dist3) # Use np.allclose to compare

    distances_for_loop(X)
    distances_vectorized(X)

    cdist(X, X)
    squareform(pdist(X))

    np.finfo(np.float64), np.finfo(np.float32), np.finfo(np.float16) # ?

    x = np.linspace(0.0, 4.0, 5).astype(np.float32)
    softmax_unstable(x)

    x = np.linspace(50.0, 90.0, 5).astype(np.float32)
    softmax_unstable(x)

    x = np.linspace(50.0, 90.0, 5).astype(np.float64)
    softmax_stable(x)

    x = np.linspace(50.0, 90.0, 5).astype(np.float64)
    softmax_unstable(x)

    x = np.array([[20.0, 20.0]])
    w = np.array([[1.0, 1.0]])
    y = np.array([1.0])

    # 1. compute logits
    scores = x @ w.T

    # 2. compute loss
    binary_cross_entropy_unstable(scores, y)

    # 1. compute logits
    scores = x @ w.T

    # 2. compute loss
    binary_cross_entropy_stable(scores, y)

    x = np.linspace(0, 30, 11).astype(np.float32)
    log_sigmoid_unstable(x)

    x = np.linspace(0, 30, 11).astype(np.float64)
    log_sigmoid_unstable(x)

    x = np.linspace(0, 30, 11).astype(np.float32)
    log_sigmoid_stable(x)

if __name__ == "__main__":
    main()
