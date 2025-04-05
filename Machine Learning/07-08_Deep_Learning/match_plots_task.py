import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Generate sample dataset
np.random.seed(42)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different network architectures
configurations = [
    ("Single hidden layer (4 units)", [4]),
    ("Single hidden layer (8 units)", [8]),
    ("Two hidden layers (4, 4)", [4, 4]),
    ("Two hidden layers (8, 4)", [8, 4]),
    ("Three hidden layers (8, 4, 2)", [8, 4, 2])
]

# Plotting
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (title, hidden_layers) in enumerate(configurations):
    # Create model
    model = Sequential()
    for units in hidden_layers:
        model.add(Dense(units, activation='relu', input_shape=(2,)))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    history = model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=0)
    
    # Plot learning curves
    axs[i].plot(history.history['accuracy'], label='Training')
    axs[i].plot(history.history['val_accuracy'], label='Validation')
    axs[i].set_title(f"Model {i+1}")
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel("Accuracy")
    axs[i].legend()
    axs[i].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.show()

# Problem Description Text
problem_text = """
**Problem: Neural Network Architectures (10 points)**

The following five plots show learning curves for different neural network architectures. Your task is to match each plot (Model 1 to Model 5) to the corresponding architecture below.

a) Single hidden layer with 4 units
b) Single hidden layer with 8 units
c) Two hidden layers with 4 units each
d) Two hidden layers with 8 and 4 units
e) Three hidden layers with 8, 4, and 2 units

✔️ Choose the correct model number (Model 1–5) for each architecture above.
"""

print(problem_text) 