import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Dropout(nn.Module):
    """
    Dropout, as discussed in the lecture and described here:
    https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
    
    Args:
        p: float, dropout probability
    """
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, input):
        """
        The module's forward pass.
        This has to be implemented for every PyTorch module.
        PyTorch then automatically generates the backward pass
        by dynamically generating the computational graph during
        execution.
        
        Args:
            input: PyTorch tensor, arbitrary shape

        Returns:
            PyTorch tensor, same shape as input
        """
        
        # TODO: Set values randomly to 0.
        if self.training:
            mask = input.new_empty(input.shape)
            mask.bernoulli_(1 - self.p)
            scaling = 1 / (1 - self.p)
            return scaling * mask * input
        else:
            return input
        
# Test dropout
test = torch.rand(10_000)
dropout = Dropout(0.2)
test_dropped = dropout(test)

# These assertions can in principle fail due to bad luck, but
# if implemented correctly they should almost always succeed.
assert np.isclose(test_dropped.mean().item(), test.mean().item(), atol=1e-2)
assert np.isclose((test_dropped > 0).float().mean().item(), 0.8, atol=1e-2)

class BatchNorm(nn.Module):
    """
    Batch normalization, as discussed in the lecture and similar to
    https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d
    
    Only uses batch statistics (no running mean for evaluation).
    Batch statistics are calculated for a single dimension.
    Gamma is initialized as 1, beta as 0.
    
    Args:
        num_features: Number of features to calculate batch statistics for.
    """
    def __init__(self, num_features):
        super().__init__()
        
        # TODO: Initialize the required parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, input):
        """
        Batch normalization over the dimension C of (N, C, L).
        
        Args:
            input: PyTorch tensor, shape [N, C, L]
            
        Return:
            PyTorch tensor, same shape as input
        """
        eps = 1e-5
        
        # TODO: Implement the required transformation
        aggregate_dims = [0, 2]
        mean = torch.mean(input, dim=aggregate_dims, keepdim=True)
        std = torch.std(input, dim=aggregate_dims, keepdim=True)
        
        input_normalized = (input - mean) / (std + eps)
        return self.gamma[None, :, None] * input_normalized + self.beta[None, :, None]

# Tests the batch normalization implementation
torch.random.manual_seed(42)
test = torch.randn(8, 2, 4)

b1 = BatchNorm(2)
test_b1 = b1(test)

b2 = nn.BatchNorm1d(2, affine=False, track_running_stats=False)
test_b2 = b2(test)

assert torch.allclose(test_b1, test_b2, rtol=0.02)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(nn.Module):
    """
    The residual block used by ResNet.
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        stride: Stride size of the first convolution, used for downsampling
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()        
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],
                                               (0, 0, 0, 0, 0, out_channels - in_channels),
                                               mode="constant", value=0))
        else:
            self.skip = nn.Sequential()
            
        # TODO: Initialize the required layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, input):
        # TODO: Execute the required layers and functions
        x1 = F.relu(self.bn1(self.conv1(input)))
        x2 = self.bn2(self.conv2(x1))
        return F.relu(x2 + self.skip(input))

class ResidualStack(nn.Module):
    """
    A stack of residual blocks.
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        stride: Stride size of the first layer, used for downsampling
        num_blocks: Number of residual blocks
    """
    
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        
        # TODO: Initialize the required layers (blocks)
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, input):
        # TODO: Execute the layers (blocks)
        x = input
        for block in self.blocks:
            x = block(x)
        return x

n = 5
num_classes = 10

# TODO: Implement ResNet via nn.Sequential
resnet = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    ResidualStack(16, 16, stride=1, num_blocks=n),
    ResidualStack(16, 32, stride=2, num_blocks=n),
    ResidualStack(32, 64, stride=2, num_blocks=n),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.squeeze()),
    nn.Linear(64, num_classes)
)

def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        
resnet.apply(initialize_weight);

class CIFAR10Subset(torchvision.datasets.CIFAR10):
    """
    Get a subset of the CIFAR10 dataset, according to the passed indices.
    """
    def __init__(self, *args, idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if idx is None:
            return
        
        self.data = self.data[idx]
        targets_np = np.array(self.targets)
        self.targets = targets_np[idx].tolist()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
])
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

ntrain = 45_000
train_set = CIFAR10Subset(root='./data', train=True, idx=range(ntrain),
                          download=True, transform=transform_train)
val_set = CIFAR10Subset(root='./data', train=True, idx=range(ntrain, 50_000),
                        download=True, transform=transform_eval)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_eval)

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=128,
                                                   shuffle=True, num_workers=2,
                                                   pin_memory=True)
dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=128,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=True)
dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=128,
                                                  shuffle=False, num_workers=2,
                                                  pin_memory=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet.to(device);

def run_epoch(model, optimizer, dataloader, train):
    """
    Run one epoch of training or evaluation.
    
    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        dataloader: Dataloader providing the data to run our model on
        train: Whether this epoch is used for training or evaluation
        
    Returns:
        Loss and accuracy in this epoch.
    """
    # TODO: Change the necessary parts to work correctly during evaluation (train=False)
    
    device = next(model.parameters()).device
    
    # Set model to training mode (for e.g. batch normalization, dropout)
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0

    # Iterate over data
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = F.cross_entropy(pred, yb)
            top1 = torch.argmax(pred, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        # statistics
        epoch_loss += loss.item()
        epoch_acc += ncorrect.item()
    
    epoch_loss /= len(dataloader.dataset)
    epoch_acc /= len(dataloader.dataset)
    return epoch_loss, epoch_acc

def fit(model, optimizer, lr_scheduler, dataloaders, max_epochs, patience):
    """
    Fit the given model on the dataset.
    
    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        lr_scheduler: Learning rate scheduler that improves training
                      in late epochs with learning rate decay
        dataloaders: Dataloaders for training and validation
        max_epochs: Maximum number of epochs for training
        patience: Number of epochs to wait with early stopping the
                  training if validation loss has decreased
                  
    Returns:
        Loss and accuracy in this epoch.
    """
    
    best_acc = 0
    curr_patience = 0
    
    for epoch in range(max_epochs):
        train_loss, train_acc = run_epoch(model, optimizer, dataloaders['train'], train=True)
        lr_scheduler.step()
        print(f"Epoch {epoch + 1: >3}/{max_epochs}, train loss: {train_loss:.2e}, accuracy: {train_acc * 100:.2f}%")
        
        val_loss, val_acc = run_epoch(model, None, dataloaders['val'], train=False)
        print(f"Epoch {epoch + 1: >3}/{max_epochs}, val loss: {val_loss:.2e}, accuracy: {val_acc * 100:.2f}%")
        
        # TODO: Add early stopping and save the best weights (in best_model_weights)
        if val_acc >= best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
                
        # Early stopping
        if epoch - best_epoch >= patience:
            break
    
    model.load_state_dict(best_model_weights)

optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# Fit model
fit(resnet, optimizer, lr_scheduler, dataloaders, max_epochs=200, patience=50)

test_loss, test_acc = run_epoch(resnet, None, dataloaders['test'], train=False)
print(f"Test loss: {test_loss:.1e}, accuracy: {test_acc * 100:.2f}%")

def main():
    # Test dropout
    test = torch.rand(10_000)
    dropout = Dropout(0.2)
    test_dropped = dropout(test)

    # These assertions can in principle fail due to bad luck, but
    # if implemented correctly they should almost always succeed.
    assert np.isclose(test_dropped.mean().item(), test.mean().item(), atol=1e-2)
    assert np.isclose((test_dropped > 0).float().mean().item(), 0.8, atol=1e-2)

    # Test batch normalization
    torch.random.manual_seed(42)
    test = torch.randn(8, 2, 4)

    b1 = BatchNorm(2)
    test_b1 = b1(test)

    b2 = nn.BatchNorm1d(2, affine=False, track_running_stats=False)
    test_b2 = b2(test)

    assert torch.allclose(test_b1, test_b2, rtol=0.02)

    # Initialize ResNet
    n = 5
    num_classes = 10

    resnet = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        ResidualStack(16, 16, stride=1, num_blocks=n),
        ResidualStack(16, 32, stride=2, num_blocks=n),
        ResidualStack(32, 64, stride=2, num_blocks=n),
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.squeeze()),
        nn.Linear(64, num_classes)
    )

    resnet.apply(initialize_weight)

    # Prepare data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    ntrain = 45_000
    train_set = CIFAR10Subset(root='./data', train=True, idx=range(ntrain),
                              download=True, transform=transform_train)
    val_set = CIFAR10Subset(root='./data', train=True, idx=range(ntrain, 50_000),
                            download=True, transform=transform_eval)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_eval)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=128,
                                                       shuffle=True, num_workers=2,
                                                       pin_memory=True)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=128,
                                                     shuffle=False, num_workers=2,
                                                     pin_memory=True)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=128,
                                                      shuffle=False, num_workers=2,
                                                      pin_memory=True)

    # Move model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    resnet.to(device)

    # Train model
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    best_model = fit(resnet, optimizer, lr_scheduler, dataloaders, max_epochs=200, patience=20)

    # Evaluate on test set
    test_loss, test_acc = run_epoch(best_model, None, dataloaders['test'], train=False)
    print(f"Test loss: {test_loss:.2e}, accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
