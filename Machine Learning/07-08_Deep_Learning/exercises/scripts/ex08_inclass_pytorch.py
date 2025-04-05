import copy
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import torch.nn.functional as F
from typeguard import typechecked

@typechecked
def sum(x: int, y: int) -> int:
    """Computes the sum of two integers.

    Args:
        x: the first integer
        y: the second integer

    Returns:
        the sum of the two integers.
    """
    return x + y

a = sum(1, 2)
print(a)

try:
    b = sum(1, 1.2)
except TypeError as e:
    print(f"TypeError: {e}")

# download MNIST dataset
mnist_dev = torchvision.datasets.MNIST("./data", train=True, download=True)
mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True)
mnist_dev, mnist_test

# check if the dataset is an instance of torch.utils.data.Dataset
isinstance(mnist_dev, torch.utils.data.Dataset)

x_dev, y_dev = mnist_dev.data, mnist_dev.targets

print(x_dev.shape, x_dev.dtype, x_dev.min(), x_dev.max())
print(y_dev.shape, y_dev.dtype, y_dev.min(), y_dev.max())

# plot the first 10 images
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax[i // 5, i % 5].imshow(x_dev[i], cmap="gray")
    ax[i // 5, i % 5].set_title(f"Label: {y_dev[i]}")
    ax[i // 5, i % 5].axis("off")
plt.tight_layout()
plt.show()

x_dev = mnist_dev.data / 255.0

# split the dataset into train and validation
NUM_TRAIN = 50_000

x_train, y_train = x_dev[:NUM_TRAIN].flatten(1), y_dev[:NUM_TRAIN]
x_val, y_val = x_dev[NUM_TRAIN:].flatten(1), y_dev[NUM_TRAIN:]

x_test = (mnist_test.data / 255.0).flatten(1)
y_test = mnist_test.targets

# check the shapes and data types
print(
    f"Training data: {x_train.shape}, {y_train.shape}, {x_train.dtype}, {y_train.dtype}"
)
print(f"Validation data: {x_val.shape}, {y_val.shape}, {x_val.dtype}, {y_val.dtype}")
print(f"Test data: {x_test.shape}, {y_test.shape}, {x_test.dtype}, {y_test.dtype}")

# define number of features and classes
NUM_FEATURES = x_train.shape[1]
NUM_CLASSES = len(torch.unique(y_train))

print(f"Num. of features: {NUM_FEATURES}")
print(f"Num. of classes: {NUM_CLASSES}")

@typechecked
def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed: the random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Show how seed affects the random number generation
print(f"Seed: 42")
set_seed(42)
print(np.random.randint(0, 100, 6))
print(np.random.randint(0, 100, 6))
set_seed(42)
print(f"Seed: 42")
print(np.random.randint(0, 100, 3))
print(np.random.randint(0, 100, 3))
print(np.random.randint(0, 100, 3))
print(np.random.randint(0, 100, 3))

SEED = 42
set_seed(SEED)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)
z = torch.tensor([7.0, 8.0, 9.0], requires_grad=True)

a = x + y
b = x + z

print(f"a.requires_grad: {a.requires_grad}")
print(f"b.requires_grad: {b.requires_grad}")

weight = torch.empty(NUM_FEATURES, NUM_CLASSES, requires_grad=False)
bias = torch.empty(NUM_CLASSES, requires_grad=False)

weight, bias

@typechecked
def sample_from_uniform(
    *size: Union[Set[int], int], low: float = -1.0, high: float = 1.0
) -> torch.Tensor:
    """Sample from uniform distribution.

    Args:
        size: the size of the output tensor
        low: the lower bound of the uniform distribution. Defaults to -1.0.
        high: the upper bound of the uniform distribution. Defaults to 1.0.

    Returns:
        the output tensor sampled from uniform distribution
    """
    # torch.rand generates samples from uniform distribution [0, 1)
    # multiply by (high - low) to get samples from uniform distribution [0, high-low)
    # shift the distribution by adding low to get samples from uniform distribution [low, high)
    return torch.rand(*size, dtype=torch.float32) * (high - low) + low


@typechecked
def xavier_uniform_init(
    weight: torch.Tensor, requires_grad: bool = True
) -> torch.Tensor:
    """Initialize weight using Xavier uniform initialization.

    Args:
        weight: the weight tensor
        requires_grad: whether the weight tensor requires gradient. Defaults to True.

    Returns:
        the weight tensor initialized using Xavier uniform initialization
    """
    fan_in, fan_out = weight.shape
    bound = np.sqrt(6 / (fan_in + fan_out))
    out = sample_from_uniform(fan_in, fan_out, low=-bound, high=bound)
    if requires_grad:
        out.requires_grad_()
    return out


@typechecked
def zeros_init(weight: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
    """Initialize weight using zeros.

    Args:
        weight: the weight tensor
        requires_grad: whether the weight tensor requires gradient. Defaults to True.

    Returns:
        the weight tensor initialized using zeros
    """
    return torch.zeros_like(weight, requires_grad=requires_grad)

# initialize weight using Glorot initialization and bias to zero
weight = xavier_uniform_init(weight, requires_grad=True)
bias = zeros_init(bias, requires_grad=True)

weight, bias

@typechecked
def model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute the logits of the model.

    Args:
        x: the input tensor
        weight: the weight tensor
        bias: the bias tensor

    Returns:
        the logits of the model
    """
    return x @ weight + bias

@typechecked
def log_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute the log softmax of the input tensor.

    Args:
        x: the input tensor

    Returns:
        the log softmax of the input tensor
    """
    return x - x.exp().sum(-1).log().unsqueeze(-1)


@typechecked
def nll_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the negative log likelihood loss.

    Args:
        logits: the logits of the model
        target: the target tensor

    Returns:
        the negative log likelihood loss
    """
    output = log_softmax(logits)
    return -output[range(target.shape[0]), target].mean()

x, label = x_train[:64], y_train[:64]
logits = model(x, weight, bias)
print(logits.shape)
loss = nll_loss(logits, label)


from torchviz import make_dot
from IPython.display import Image

logits = model(x, weight, bias)
loss = nll_loss(logits, label)
params = {"weight": weight, "bias": bias}

dot = make_dot(loss, params=params, show_attrs=True, show_saved=True)
dot.render(
    "computational_graph", directory="img", format="png"
)  # Save the graph as a PNG file

Image("img/computational_graph.png", width=600)

# define loss function
criterion = nll_loss

# define hyperparameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 5
BATCH_SIZE = 64


@typechecked
def fit(
    model: Callable,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    criterion: Callable,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> None:
    """Train the model.

    Args:
        model: a callable that returns the logits given the input and the parameters
        weight: the weight tensor
        bias: the bias tensor
        x_train: the training features tensor
        y_train: the training labels tensor
        criterion: a callable that returns the loss given the logits and the labels
        learning_rate: the learning rate
        num_epochs: the number of epochs
        batch_size: the batch size
    """
    # Compute number of batches
    num_train = x_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))

    # Initialize loss and accuracy history
    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        # Initialize epoch loss and accuracy
        epoch_loss = 0.0
        epoch_acc = 0.0

        for i in range(num_batches):
            # Get mini-batch
            start_i = i * batch_size
            end_i = min(start_i + batch_size, num_train)
            x_batch, y_batch = x_train[start_i:end_i], y_train[start_i:end_i]

            # Generate predictions
            pred = model(x_batch, weight, bias)

            # Calculate loss
            loss = criterion(pred, y_batch)

            # Zero out the gradients
            if weight.grad is not None:
                weight.grad.zero_()
            if bias.grad is not None:
                bias.grad.zero_()

            # Compute gradients
            loss.backward()

            # Update parameters
            with torch.inference_mode():  # "new version" of torch.no_grad() (potentially faster)
                weight -= learning_rate * weight.grad
                bias -= learning_rate * bias.grad

            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            top1 = pred.argmax(-1) == y_batch
            ncorrect = torch.sum(top1).item()
            epoch_acc += ncorrect

        # Epoch loss and accuracy
        epoch_loss /= num_train
        epoch_acc /= num_train

        # Print progress
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
        )

        # Record loss and accuracy
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

    # Plot loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_history)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(acc_history)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    fig.suptitle(f"Training")
    plt.show()

fit(
    model,
    weight,
    bias,
    x_train,
    y_train,
    criterion,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
)

criterion = nll_loss
pred = model(x_train[:BATCH_SIZE], weight, bias)
target = y_train[:BATCH_SIZE]
print(criterion(pred, target))

criterion = F.cross_entropy
pred = model(x_train[:BATCH_SIZE], weight, bias)
target = y_train[:BATCH_SIZE]
print(criterion(pred, target))

from torch import nn


@typechecked
class LogisticRegression(nn.Module):
    """Logistic regression model.

    The model consists of a single linear layer that maps from the number of features to the number of classes.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        """Constructor method for LogisticRegression.

        Args:
            num_features: the number of features
            num_classes: the number of classes
        """
        super().__init__()

        # define the parameters
        self.weight = nn.Parameter(torch.Tensor(num_features, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        # initialize the parameters
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the parameters.

        The weight is initialized using Xavier uniform initialization and the bias is initialized to zero.
        """
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: the input tensor

        Returns:
            the logits
        """
        return x @ self.weight + self.bias

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)

@typechecked
def fit(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    criterion: Callable,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        x_train: the training features tensor
        y_train: the training labels tensor
        criterion: a callable that returns the loss given the logits and the labels
        learning_rate: the learning rate
        num_epochs: the number of epochs
        batch_size: the batch size
    """
    num_train = x_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))

    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for i in range(num_batches):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, num_train)
            x_batch = x_train[start_i:end_i]
            y_batch = y_train[start_i:end_i]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            # Zero out the gradients
            model.zero_grad()

            # Compute gradients
            loss.backward()

            # Update parameters
            with torch.inference_mode():
                for param in model.parameters():
                    param -= learning_rate * param.grad

            epoch_loss += loss.item()
            top1 = pred.argmax(-1) == y_batch
            ncorrect = torch.sum(top1).item()
            epoch_acc += ncorrect

        epoch_loss /= num_train
        epoch_acc /= num_train

        print_epoch_summary(epoch, num_epochs, epoch_loss, epoch_acc)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

    plot_curves(loss_history, acc_history, "Training")


@typechecked
def print_epoch_summary(epoch: int, num_epochs: int, loss: float, acc: float) -> None:
    """Print the epoch summary.

    The summary includes the epoch number, the number of epochs, the loss, and the accuracy.

    Args:
        epoch: the epoch number
        num_epochs: the number of epochs
        loss: the loss
        acc: the accuracy
    """
    print(
        f"Epoch {epoch+1:>{len(str(num_epochs))}}/{num_epochs} | Loss: {loss:.4f}"
        + f" | Accuracy: {acc:.4f}"
    )


@typechecked
def plot_curves(losses: list, accuracies: list, mode: str) -> None:
    """Plot the loss and accuracy curves.

    It plots the loss curve on the left and the accuracy curve on the right via matplotlib.

    Args:
        losses: the list of losses
        accuracies: the list of accuracies
        mode: the mode (e.g., training, validation, or test)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(accuracies)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    fig.suptitle(f"{mode}")
    plt.show()

fit(model, x_train, y_train, criterion, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE)

@typechecked
class LogisticRegression(nn.Module):
    """Logistic regression model.

    The model consists of a single linear layer that maps from the number of features to the number of classes.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        """Constructor method for LogisticRegression.

        Args:
            num_features: the number of features
            num_classes: the number of classes
        """
        super().__init__()
        # define the linear layer
        self.lin = nn.Linear(num_features, num_classes, bias=True)

        # initialize the weights
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.

        The weight is initialized using Xavier uniform initialization and the bias is initialized to zero.
        """
        if isinstance(self.lin, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(self.lin.weight)
            if self.lin.bias is not None:
                nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: the input tensor

        Returns:
            the logits
        """
        return self.lin(x)

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)
criterion = F.cross_entropy
fit(model, x_train, y_train, criterion, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE)

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)
criterion = F.cross_entropy

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

@typechecked
def fit(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        x_train: the training features tensor
        y_train: the training labels tensor
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
        batch_size: the batch size
    """
    num_train = x_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))

    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for i in range(num_batches):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, num_train)
            x_batch = x_train[start_i:end_i]
            y_batch = y_train[start_i:end_i]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            # Zero out the gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Optimization step
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += torch.sum(pred.argmax(-1) == y_batch).item()

        epoch_loss /= num_train
        epoch_acc /= num_train

        print_epoch_summary(epoch, num_epochs, epoch_loss, epoch_acc)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

    plot_curves(loss_history, acc_history, "Training")

fit(model, x_train, y_train, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE)

from torch.utils.data import TensorDataset

# create Tensor datasets
train_dataset = TensorDataset(x_train, y_train)

# check if the dataset is an instance of torch.utils.data.Dataset
isinstance(train_dataset, torch.utils.data.Dataset)

@typechecked
def fit(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        train_dataset: the training dataset (an instance of torch.utils.data.Dataset)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
        batch_size: the batch size
    """
    num_train = x_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))

    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for i in range(num_batches):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, num_train)
            x_batch, y_batch = train_dataset[start_i:end_i]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += torch.sum(pred.argmax(-1) == y_batch).item()

        epoch_loss /= num_train
        epoch_acc /= num_train

        print_epoch_summary(epoch, num_epochs, epoch_loss, epoch_acc)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

    plot_curves(loss_history, acc_history, "Training")

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

fit(model, train_dataset, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE)

mnist_dev = torchvision.datasets.MNIST("./data", train=True, download=True)
mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True)

print(len(mnist_dev))
print(mnist_dev[0])

mnist_dev = torchvision.datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_test = torchvision.datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_dev, mnist_test

image, label = mnist_dev[0]
image.shape, label

@typechecked
class FlattenTransform:
    """Data transform that flattens the input tensor"""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flatten the input tensor.

        Args:
            image: the input tensor

        Returns:
            the flattened input tensor
        """
        return tensor.view(-1)

@typechecked
class IntToTensorTransform:
    """Data transform that converts the input integer to a tensor"""

    def __call__(self, value: int) -> torch.Tensor:
        """Convert the input integer to a tensor.

        Args:
            value: the input integer

        Returns:
            the converted tensor
        """
        return torch.tensor(value, dtype=torch.int64)

mnist_dev = torchvision.datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            FlattenTransform(),
        ]
    ),
    target_transform=IntToTensorTransform(),
)

mnist_test = torchvision.datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            FlattenTransform(),
        ]
    ),
    target_transform=IntToTensorTransform(),
)

mnist_dev, mnist_test

image, label = mnist_dev[0]
image.shape, label

mnist_train, mnist_val = torch.utils.data.random_split(mnist_dev, [50000, 10000])
mnist_train, mnist_val

image, label = mnist_train[0]
len(mnist_train), image.shape, label

print(
    f"Train | Num. of samples: {len(mnist_train)}, X shape: {mnist_train[0][0].shape}"
)
print(f"Val   | Num. of samples: {len(mnist_val)}, X shape: {mnist_val[0][0].shape}")
print(f"Test  | Num. of samples: {len(mnist_test)}, X shape: {mnist_test[0][0].shape}")

from torch.utils.data import DataLoader

# create DataLoader
train_dataloader = DataLoader(
    dataset=mnist_test,
    batch_size=BATCH_SIZE,
    generator=torch.Generator().manual_seed(SEED),  # this ensures reproducibility
)

# check if the dataset is an instance of torch.utils.data.DataLoader
isinstance(train_dataloader, torch.utils.data.DataLoader)

# iterate over the DataLoader
for x_batch, y_batch in train_dataloader:
    print(x_batch.shape, y_batch.shape)
    break

@typechecked
def fit(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        train_dataloader: the training dataloader (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
    """
    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for x_batch, y_batch in train_dataloader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += torch.sum(pred.argmax(-1) == y_batch).item()

        epoch_loss /= len(train_dataloader.dataset)
        epoch_acc /= len(train_dataloader.dataset)

        print_epoch_summary(epoch, num_epochs, epoch_loss, epoch_acc)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

    plot_curves(loss_history, acc_history, "Training")

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

fit(model, train_dataloader, criterion, optimizer, NUM_EPOCHS)

dataloaders = {}
dataloaders["train"] = DataLoader(
    mnist_train,
    batch_size=BATCH_SIZE,
    shuffle=False,
    generator=torch.Generator().manual_seed(SEED),
)
dataloaders["val"] = DataLoader(
    mnist_val,
    batch_size=2 * BATCH_SIZE,
    shuffle=False,
    generator=torch.Generator().manual_seed(SEED),
)

@typechecked
def fit(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        dataloaders: the dictionary of dataloaders (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
    """
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        epoch_loss = {"train": 0.0, "val": 0.0}
        epoch_acc = {"train": 0.0, "val": 0.0}

        # Training
        model.train()
        for x_batch, y_batch in dataloaders["train"]:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss["train"] += loss.item()
            epoch_acc["train"] += torch.sum(pred.argmax(-1) == y_batch).item()

        epoch_loss["train"] /= len(dataloaders["train"].dataset)
        epoch_acc["train"] /= len(dataloaders["train"].dataset)

        # Validation
        model.eval()
        for x_batch, y_batch in dataloaders["val"]:
            with torch.inference_mode():
                pred = model(x_batch)
                loss = criterion(pred, y_batch)

            epoch_loss["val"] += loss.item()
            epoch_acc["val"] += torch.sum(pred.argmax(-1) == y_batch).item()

        epoch_loss["val"] /= len(dataloaders["val"].dataset)
        epoch_acc["val"] /= len(dataloaders["val"].dataset)

        print_epoch_summary(epoch, num_epochs, epoch_loss, epoch_acc)

        loss_history["train"].append(epoch_loss["train"])
        loss_history["val"].append(epoch_loss["val"])
        acc_history["train"].append(epoch_acc["train"])
        acc_history["val"].append(epoch_acc["val"])

    training_history = {"loss": loss_history["train"], "acc": acc_history["train"]}
    validation_history = {"loss": loss_history["val"], "acc": acc_history["val"]}
    plot_curves(training_history, validation_history)


@typechecked
def print_epoch_summary(
    epoch: int, num_epochs: int, loss: Dict[str, float], acc: Dict[str, float]
) -> None:
    """Print the epoch summary.

    The summary includes the epoch number, the number of epochs, the loss, and the accuracy.

    Args:
        epoch: the epoch number
        num_epochs: the number of epochs
        loss: the loss
        acc: the accuracy
    """
    print(
        f"Epoch {epoch+1:>{len(str(num_epochs))}}/{num_epochs} | "
        f"Train - loss: {loss['train']:.4f}, acc: {acc['train']:.4f} | "
        f"Val - loss: {loss['val']:.4f}, acc: {acc['val']:.4f}"
    )


@typechecked
def plot_curves(
    training_history: Dict[str, list], validation_history: Dict[str, list]
) -> None:
    """Plot the loss and accuracy curves.

    It plots the loss curve on the left and the accuracy curve on the right via matplotlib.

    Args:
        training_history: the training history
        validation_history: the validation history
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(training_history["loss"], label="Training")
    ax1.plot(validation_history["loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(training_history["acc"], label="Training")
    ax2.plot(validation_history["acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.show()

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

fit(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

@typechecked
def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = False,
) -> Tuple[float, float]:
    """Run one epoch.

    It runs one epoch of training, validation, or test, and returns the loss and accuracy. If training is True, it also updates the parameters.

    Args:
        model: the model (an instance of torch.nn.Module)
        dataloader: the dataloader (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer). Defaults to None.
        train: whether to train the model. Defaults to False.

    Returns:
        the loss and accuracy
    """
    epoch_loss = 0.0
    epoch_acc = 0.0

    if train:
        model.train()
    else:
        model.eval()

    for x_batch, y_batch in dataloader:
        with torch.set_grad_enabled(train):
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += (pred.argmax(-1) == y_batch).sum().item()

    epoch_loss /= len(dataloader.dataset)
    epoch_acc /= len(dataloader.dataset)

    return epoch_loss, epoch_acc

@typechecked
def fit(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        dataloaders: the dictionary of dataloaders (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
    """
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = run_epoch(
            model, dataloaders["train"], criterion, optimizer, train=True
        )

        # Validation
        val_loss, val_acc = run_epoch(model, dataloaders["val"], criterion, train=False)

        loss = {"train": train_loss, "val": val_loss}
        acc = {"train": train_acc, "val": val_acc}
        print_epoch_summary(epoch, num_epochs, loss, acc)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        acc_history["train"].append(train_acc)
        acc_history["val"].append(val_acc)

    training_history = {"loss": loss_history["train"], "acc": acc_history["train"]}
    validation_history = {"loss": loss_history["val"], "acc": acc_history["val"]}
    plot_curves(training_history, validation_history)

set_seed(SEED)
model = LogisticRegression(NUM_FEATURES, NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

fit(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

test_dataloader = DataLoader(
    dataset=mnist_test,
    batch_size=2 * BATCH_SIZE,
    shuffle=False,
    generator=torch.Generator().manual_seed(SEED),
)

test_loss, test_acc = run_epoch(model, test_dataloader, criterion, train=False)
print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

@typechecked
class CNN(nn.Module):
    """Convolutional neural network model."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Constructor method for CNN.

        Args:
            num_channels: the number of channels
            num_classes: the number of classes
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the parameters.

        The weight is initialized using Xavier uniform initialization and the bias is initialized to zero.
        """
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: the input tensor

        Returns:
            the logits
        """
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 4)
        return x.view(-1, x.size(1))

NUM_CHANNELS = 16
LEARNING_RATE = 0.1
MOMENTUM = 0.9

set_seed(SEED)
model = CNN(num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

fit(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)

run_epoch(model, test_dataloader, criterion, train=False)

# check if GPU is available
print(torch.cuda.is_available())

# set the device to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# check model device
print(model.conv1.weight.device)
model.to(device)
print(model.conv1.weight.device)

NUM_WORKERS = 0

dataloaders = {}
dataloaders["train"] = DataLoader(
    mnist_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    generator=torch.Generator().manual_seed(SEED),
)
dataloaders["val"] = DataLoader(
    mnist_val,
    batch_size=2 * BATCH_SIZE,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    generator=torch.Generator().manual_seed(SEED),
)
dataloaders["test"] = DataLoader(
    mnist_test,
    batch_size=2 * BATCH_SIZE,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    generator=torch.Generator().manual_seed(SEED),
)

def instantiate_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Instantiate dataloaders.

    Args:
        train_dataset: the training dataset
        val_dataset: the validation dataset
        test_dataset: the test dataset
        batch_size: the batch size
        num_workers: the number of workers
        seed: the seed

    Returns:
        the dictionary of dataloaders
    """
    dataloaders = {}
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=2 * batch_size,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    return dataloaders

import multiprocessing

num_cores = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cores)

for x_batch, y_batch in dataloaders["train"]:
    pass

@typechecked
def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: torch.device = torch.device("cpu"),
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = False,
) -> Tuple[float, float]:
    """Run one epoch.

    It runs one epoch of training, validation, or test, and returns the loss and accuracy. If training is True, it also updates the parameters.

    Args:
        model: the model (an instance of torch.nn.Module)
        dataloader: the dataloader (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        device: the device (cpu or gpu). Defaults to torch.device("cpu").
        optimizer: the optimizer (an instance of torch.optim.Optimizer). Defaults to None.
        train: whether to train the model. Defaults to False.

    Returns:
        the loss and accuracy
    """
    epoch_loss = 0.0
    epoch_acc = 0.0

    # Move model to the device
    model = model.to(device)

    if train:
        model.train()
    else:
        model.eval()

    for x_batch, y_batch in dataloader:
        # Move data to the device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.set_grad_enabled(train):
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += (pred.argmax(-1) == y_batch).sum().item()

    epoch_loss /= len(dataloader.dataset)
    epoch_acc /= len(dataloader.dataset)

    return epoch_loss, epoch_acc


@typechecked
def fit(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        dataloaders: the dictionary of dataloaders (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
        device: the device (cpu or gpu). Defaults to torch.device("cpu").
    """
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = run_epoch(
            model, dataloaders["train"], criterion, device, optimizer, train=True
        )

        # Validation
        val_loss, val_acc = run_epoch(
            model, dataloaders["val"], criterion, device, train=False
        )

        loss = {"train": train_loss, "val": val_loss}
        acc = {"train": train_acc, "val": val_acc}
        print_epoch_summary(epoch, num_epochs, loss, acc)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        acc_history["train"].append(train_acc)
        acc_history["val"].append(val_acc)

    training_history = {"loss": loss_history["train"], "acc": acc_history["train"]}
    validation_history = {"loss": loss_history["val"], "acc": acc_history["val"]}
    plot_curves(training_history, validation_history)

set_seed(SEED)
model = CNN(num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
dataloaders = instantiate_dataloaders(
    mnist_train, mnist_val, mnist_test, BATCH_SIZE, NUM_WORKERS, SEED
)

fit(model, dataloaders, criterion, optimizer, NUM_EPOCHS, device)

test_loss, test_acc = run_epoch(model, dataloaders["test"], criterion, train=False)
print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

@typechecked
class Lambda(nn.Module):
    """A module that applies a given function."""

    def __init__(self, func: Callable[[torch.Tensor], Any]) -> None:
        """Constructor method for Lambda.

        Args:
            func: the function to apply
        """
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass.

        Args:
            x: the input tensor

        Returns:
            the output of the function
        """
        return self.func(x)

set_seed(SEED)
model = nn.Sequential(
    Lambda(lambda x: x.view(-1, 1, 28, 28)),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

@typechecked
def initialize_weights(m: nn.Module) -> None:
    """Initialize the weights.

    The weight is initialized using Xavier uniform initialization and the bias is initialized to zero.

    Args:
        m: the module (an instance of torch.nn.Module)
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


model.apply(initialize_weights)

criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
dataloaders = instantiate_dataloaders(
    mnist_train, mnist_val, mnist_test, BATCH_SIZE, NUM_WORKERS, SEED
)

fit(model, dataloaders, criterion, optimizer, NUM_EPOCHS, device=device)

test_loss, test_acc = run_epoch(model, dataloaders["test"], criterion, train=False)
print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

@typechecked
class CNN(nn.Module):
    """Convolutional neural network model."""

    def __init__(self, num_layers: int, num_channels: int, num_classes: int) -> None:
        """Constructor method for CNN.

        Args:
            num_layers: the number of layers
            num_channels: the number of channels
            num_classes: the number of classes
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = 1 if i == 0 else num_channels
            out_channels = num_channels if i < num_layers - 1 else num_classes
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.layers.append(nn.ReLU())
        self.layers.append(nn.AdaptiveAvgPool2d(1))

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the parameters.

        The weight is initialized using Xavier uniform initialization and the bias is initialized to zero.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: the input tensor

        Returns:
            the logits
        """
        x = x.view(-1, 1, 28, 28)
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return x

set_seed(SEED)
NUM_LAYERS = 3
model = CNN(num_layers=NUM_LAYERS, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
dataloaders = instantiate_dataloaders(
    mnist_train, mnist_val, mnist_test, BATCH_SIZE, NUM_WORKERS, SEED
)

fit(model, dataloaders, criterion, optimizer, NUM_EPOCHS, device=device)

test_loss, test_acc = run_epoch(model, dataloaders["test"], criterion, train=False)
print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

type(model.state_dict()), len(
    model.state_dict()
), model.state_dict().keys(), model.state_dict()["layers.0.weight"].shape

@typechecked
def fit(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """Train the model.

    Args:
        model: the model (an instance of torch.nn.Module)
        dataloaders: the dictionary of dataloaders (an instance of torch.utils.data.DataLoader)
        criterion: a callable that returns the loss given the logits and the labels
        optimizer: the optimizer (an instance of torch.optim.Optimizer)
        num_epochs: the number of epochs
        patience: the patience for early stopping
        device: the device (cpu or gpu)

    Returns:
        the state dict of the best model
    """
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}

    best_val_acc = 0.0
    curr_patience = patience

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = run_epoch(
            model, dataloaders["train"], criterion, device, optimizer, train=True
        )

        # Validation
        val_loss, val_acc = run_epoch(
            model, dataloaders["val"], criterion, device, train=False
        )

        loss = {"train": train_loss, "val": val_loss}
        acc = {"train": train_acc, "val": val_acc}
        print_epoch_summary(epoch, num_epochs, loss, acc)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        acc_history["train"].append(train_acc)
        acc_history["val"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            curr_patience = patience
            # Save the best model state dict as a checkpoint
            ckpt = copy.deepcopy(model.state_dict())
            # Save the best model to disk
            torch.save(ckpt, "ckpt.pt")
        else:
            curr_patience -= 1
            if curr_patience == 0:
                print("Early stopping")
                break

    training_history = {"loss": loss_history["train"], "acc": acc_history["train"]}
    validation_history = {"loss": loss_history["val"], "acc": acc_history["val"]}
    plot_curves(training_history, validation_history)

    return ckpt

set_seed(SEED)
PATIENCE = 3
model = CNN(num_layers=3, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
dataloaders = instantiate_dataloaders(
    mnist_train, mnist_val, mnist_test, BATCH_SIZE, NUM_WORKERS, SEED
)

best_model_state_dict = fit(
    model, dataloaders, criterion, optimizer, NUM_EPOCHS, PATIENCE, device=device
)

# load the model from state dict
model.load_state_dict(best_model_state_dict)

test_loss, test_acc = run_epoch(model, dataloaders["test"], criterion, train=False)
print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")
