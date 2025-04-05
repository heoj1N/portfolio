import numpy as np

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from scipy.special import softmax

# # In-class exercise 7: Deep Learning 1 (Part B) - Backprop

class Add:
    def forward(self, x, y):
        return x + y

    def backward(self, d_out):
        d_x = d_out
        d_y = d_out
        return d_x, d_y
    
class Multiply:
    def forward(self, x, y):
        self.cache = (x, y)
        return x * y

    def backward(self, d_out):
        x, y = self.cache
        d_x = d_out * y
        d_y = d_out * x
        return d_x, d_y
    
class Sum:
    def forward(self, x):
        self.cache = x
        return np.sum(x)

    def backward(self, d_out):
        x = self.cache
        d_x = d_out * np.ones_like(x) # !
        return d_x

class DotProduct:
    def forward(self, x, y):
        self.cache = x, y
        return np.dot(x, y)

    def backward(self, d_out):
        x, y = self.cache
        d_x = d_out * y
        d_y = d_out * x
        return d_x, d_y

class Affine:
    def forward(self, inputs, weight, bias):
        """Forward pass of an affine (fully connected) layer.

        Args:
            inputs: input matrix, shape (N, D)
            weight: weight matrix, shape (D, H)
            bias: bias vector, shape (H)

        Returns
            out: output matrix, shape (N, H)
        """
        self.cache = (inputs, weight, bias)
        out = inputs @ weight + bias # out = inputs.dot(weight) + bias

        assert out.shape[0] == inputs.shape[0]
        assert out.shape[1] == weight.shape[1] == bias.shape[0]
        return out

    def backward(self, d_out):
        """Backward pass of an affine (fully connected) layer.

        Args:
            d_out: incoming derivaties, shape (N, H)

        Returns:
            d_inputs: gradient w.r.t. the inputs, shape (N, D)
            d_weight: gradient w.r.t. the weight, shape (D, H)
            d_bias: gradient w.r.t. the bias, shape (H)
        """
        inputs, weight, bias = self.cache

        d_inputs = d_out @ weight.T
        d_weight = inputs.T @ d_out
        d_bias = d_out.sum(axis=0)

        assert np.all(d_inputs.shape == inputs.shape)
        assert np.all(d_weight.shape == weight.shape)
        assert np.all(d_bias.shape == bias.shape)
        return d_inputs, d_weight, d_bias

class CategoricalCrossEntropy:
    def forward(self, logits, labels):
        """Compute categorical cross-entropy loss.

        Args:
            logits: class logits, shape (N, K)
            labels: target labels in one-hot format, shape (N, K)

        Returns:
            loss: loss value, float (a single number)
        """
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
        log_probs = logits_shifted - log_sum_exp
        N = labels.shape[0]
        loss = -np.sum(labels * log_probs) / N

        probs = np.exp(log_probs)

        # probs is the (N, K) matrix of class probabilities
        self.cache = (probs, labels)
        assert isinstance(loss, float)
        return loss

    def backward(self, d_out=1.0):
        """Backward pass of the Cross Entropy loss.

        Args:
            d_out: Incoming derivatives. We set this value to 1.0 by default,
                since this is the terminal node of our computational graph
                (i.e. we usually want to compute gradients of loss w.r.t.
                other model parameters).

        Returns:
            d_logits: gradient w.r.t. the logits, shape (N, K)
            d_labels: gradient w.r.t. the labels
                we don't need d_labels for our models, so we don't
                compute it and set it to None. It's only included in the
                function definition for consistency with other layers.
        """
        probs, labels = self.cache

        N = labels.shape[0]
        d_logits = d_out * (probs - labels) / N

        d_labels = None
        assert np.all(d_logits.shape == probs.shape == labels.shape)
        return d_logits, d_labels

class LogisticRegression:
    """Logistic regression model.

    Gradients are computed with backpropagation.
    """

    def __init__(self, num_features, num_classes, learning_rate=1e-2):
        self.learning_rate = learning_rate

        # Initialize the model parameters
        self.params = {
            "W": np.zeros([num_features, num_classes]),
            "b": np.zeros([num_classes]),
        }

        # Define layers
        self.affine = Affine()
        self.cross_entropy = CategoricalCrossEntropy()

    def predict(self, X):
        """Generate predictions for one minibatch.

        Args:
            X: data matrix, shape (N, D)

        Returns:
            Y_pred: predicted class probabilities, shape (N, D)
            Y_pred[n, k] = probability that sample n belongs to class k
        """
        logits = self.affine.forward(X, self.params["W"], self.params["b"])
        return softmax(logits, axis=1)

    def step(self, X, Y):
        """Perform one step of gradient descent on the minibatch of data."""
        # Forward pass - compute the loss on training data
        logits = self.affine.forward(X, self.params["W"], self.params["b"])
        loss = self.cross_entropy.forward(logits, Y)

        # Backward pass - compute the gradients of loss w.r.t. all the model parameters
        grads = {}
        d_logits, _ = self.cross_entropy.backward()
        _, grads["W"], grads["b"] = self.affine.backward(d_logits)

        # Apply the gradients
        for p in self.params:
            self.params[p] = self.params[p] - self.learning_rate * grads[p]
        return loss

def predict(X, W, b):
    """Generate predictions for a multi-class logistic regression model.

    Args:
        X: data matrix, shape (N, D)
        W: weight matrix, shape (D, K)
        b: bias vector, shape (K)

    Returns:
        Y_pred: Predicted class probabilities, shape (N, K).
            Y_pred[n, k] = probability that sample n belongs to class k.
    """
    return softmax(X @ W + b, axis=1)

def nll_loss(X, W, b, Y):
    """Compute negative log-likelihood of a logistic regression model.

    Also known as categorical cross entropy loss.

    Args:
        X: data matrix, shape (N, D)
        W: weight matrix, shape (D, K)
        b: bias vector, shape (K)
        Y: true labels in one-hot format, shape (N, K)

    Returns:
        loss: loss of the logistic regression model, shape ()
    """
    N = X.shape[0]
    logits = X @ W + b

    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    loss = -np.sum(Y * log_probs) / N
    return loss

def nll_grad(X, W, b, Y):
    """Compute gradient of the NLL loss w.r.t. W and b.

    Args:
        X: data matrix, shape (N, D)
        W: weight matrix, shape (D, K)
        b: bias vector, shape (K)
        Y: true labels in one-hot format, shape (N, K)

    Returns:
        d_W: gradient of the los w.r.t. W, shape (D, K)
        d_b: gradient of the los w.r.t. b, shape (K)
    """
    N = X.shape[0]
    probas = softmax(X @ W + b, axis=1) - Y
    d_W = X.T @ probas / N
    d_b = probas.sum(axis=0) / N
    return d_W, d_b

def main():
    X, y = load_digits(return_X_y=True)
    Y = label_binarize(y, classes=np.unique(y)) # Convert labels into one-hot format
    K = Y.shape[1]  # number of classes
    D = X.shape[1]  # number of features

    np.random.seed(123)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Dataset consists of 1797 samples. 
    # Each sample is represented with a 64-dimensional vector and belongs to one of 10 classes.
    X.shape, Y.shape


    x = np.arange(1, 5, dtype=np.float32)
    y = np.arange(-1, 3, dtype=np.float32)

    mult = Multiply()
    vec_sum = Sum()

    w = mult.forward(x, y)
    z = vec_sum.forward(w)

    d_w = vec_sum.backward(1.0)
    d_x, d_y = mult.backward(d_w)

    z, d_x, d_y


    x = np.arange(1, 5, dtype=np.float32)
    y = np.arange(-1, 3, dtype=np.float32)

    dp = DotProduct()
    z = dp.forward(x, y)
    d_x, d_y = dp.backward(1.0)

    z, d_x, d_y


    # Initialize learnable model parameters
    W = np.zeros([D, K])
    b = np.zeros([K])

    # Specify optimization parameters
    learning_rate = 1e-2
    max_epochs = 301
    report_frequency = 25

    for epoch in range(max_epochs):
        # Compute train loss
        loss = nll_loss(X_train, W, b, Y_train)

        if epoch % report_frequency == 0:
            print(f"Epoch {epoch:4d}, loss = {loss:.4f}")

        # Perform the update
        d_W, d_b = nll_grad(X_train, W, b, Y_train)

        W = W - learning_rate * d_W
        b = b - learning_rate * d_b

    loss_test = nll_loss(X_test, W, b, Y_test)
    y_pred_test = predict(X_test, W, b).argmax(axis=1)
    y_test = Y_test.argmax(axis=1)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"loss_test = {loss_test:.4f}, accuracy_test = {acc_test:.3f}")


    # Specify optimization parameters
    learning_rate = 1e-2
    max_epochs = 301
    report_frequency = 25

    log_reg = LogisticRegression(num_features=D, num_classes=K, learning_rate=learning_rate)

    for epoch in range(max_epochs):
        loss = log_reg.step(X_train, Y_train)

        if epoch % report_frequency == 0:
            print(f"Epoch {epoch:4d}, loss = {loss:.4f}")

    y_pred_test = log_reg.predict(X_test).argmax(axis=1)
    loss_test = log_reg.step(X_test, Y_test)
    y_test = Y_test.argmax(axis=1)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"loss_test = {loss_test:.4f}, accuracy_test = {acc_test:.4f}")

if __name__ == "__main__":
    main()
