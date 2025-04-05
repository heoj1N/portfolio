import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_2d_visualization():
    # Create a grid of points
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate different norms
    Z1 = np.abs(X) + np.abs(Y)  # L1 norm (Manhattan)
    Z2 = np.sqrt(X**2 + Y**2)   # L2 norm (Euclidean)
    Z3 = np.cbrt(np.abs(X)**3 + np.abs(Y)**3)  # L3 norm
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('2D Distance Norms Visualization', fontsize=16)
    
    # Plot each norm
    norms = [(Z1, 'L1 Norm (Manhattan)'), (Z2, 'L2 Norm (Euclidean)'), (Z3, 'L3 Norm')]
    for ax, (Z, title) in zip(axes, norms):
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    plt.savefig('2d_distance_norms.png')
    plt.close()

def create_3d_visualization():
    # Create a grid of points
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculate different norms
    Z1 = np.abs(X) + np.abs(Y)  # L1 norm (Manhattan)
    Z2 = np.sqrt(X**2 + Y**2)   # L2 norm (Euclidean)
    Z3 = np.cbrt(np.abs(X)**3 + np.abs(Y)**3)  # L3 norm
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('3D Distance Norms Visualization', fontsize=16)
    
    # Plot each norm
    norms = [(Z1, 'L1 Norm (Manhattan)'), (Z2, 'L2 Norm (Euclidean)'), (Z3, 'L3 Norm')]
    for i, (Z, title) in enumerate(norms, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Distance')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('3d_distance_norms.png')
    plt.close()

if __name__ == "__main__":
    # Create both visualizations
    create_2d_visualization()
    create_3d_visualization()
    print("Visualizations have been created: '2d_distance_norms.png' and '3d_distance_norms.png'") 