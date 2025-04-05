import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def print_matrix_info(matrix, name):
    """Print matrix information in a formatted way."""
    print(f"\n{name} Matrix:")
    print(f"Shape: {matrix.shape}")
    print(f"First 5 rows:\n{matrix[:5]}")
    print(f"Mean: {np.mean(matrix, axis=0)}")
    print(f"Standard Deviation: {np.std(matrix, axis=0)}")
    if matrix.shape[0] == matrix.shape[1]:  # Only calculate determinant for square matrices
        print(f"Determinant: {np.linalg.det(matrix):.4f}")
        print(f"Trace: {np.trace(matrix):.4f}")

def plot_data_and_pcs(X, mean, eigenvectors, eigenvalues, iteration=None):
    """Plot data points and principal components."""
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data Points')
    
    # Plot mean
    plt.scatter(mean[0], mean[1], c='red', s=100, label='Mean')
    
    # Plot principal components
    for i, (eigenvector, eigenvalue) in enumerate(zip(eigenvectors.T, eigenvalues)):
        plt.arrow(mean[0], mean[1],
                 eigenvector[0] * np.sqrt(eigenvalue),
                 eigenvector[1] * np.sqrt(eigenvalue),
                 head_width=0.1, head_length=0.1, fc='red', ec='red',
                 label=f'PC{i+1} (λ={eigenvalue:.2f})')
    
    plt.title(f'PCA Visualization{" - Iteration " + str(iteration) if iteration else ""}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def pca_step_by_step(X, n_components=2):
    """Perform PCA step by step with detailed output."""
    print("="*50)
    print("PCA Step-by-Step Demonstration")
    print("="*50)
    
    # Step 1: Standardize the data
    print("\nStep 1: Standardizing the data")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    print_matrix_info(X_std, "Standardized Data")
    
    # Step 2: Calculate covariance matrix
    print("\nStep 2: Calculating covariance matrix")
    cov_matrix = np.cov(X_std.T)
    print_matrix_info(cov_matrix, "Covariance")
    
    # Step 3: Eigendecomposition
    print("\nStep 3: Eigendecomposition")
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\nEigenvalues:")
    for i, val in enumerate(eigenvalues):
        print(f"λ{i+1}: {val:.4f}")
    
    print("\nEigenvectors:")
    for i, vec in enumerate(eigenvectors.T):
        print(f"v{i+1}: {vec}")
    
    # Step 4: Project data onto principal components
    print("\nStep 4: Projecting data onto principal components")
    X_pca = X_std.dot(eigenvectors)
    print_matrix_info(X_pca, "Projected Data")
    
    # Calculate explained variance
    explained_variance = eigenvalues / np.sum(eigenvalues)
    print("\nExplained Variance:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    # Plot results
    plot_data_and_pcs(X_std, np.mean(X_std, axis=0), eigenvectors, eigenvalues)
    
    return X_pca, eigenvalues, eigenvectors, scaler, X_std

def reconstruct_data(X_pca, eigenvectors, scaler, n_components=None):
    """Reconstruct original data from PCA projections."""
    print("\nStep 5: Reconstructing data from PCA projections")
    
    if n_components is None:
        n_components = X_pca.shape[1]
    
    # Use only the first n_components
    X_pca_reduced = X_pca[:, :n_components]
    eigenvectors_reduced = eigenvectors[:, :n_components]
    
    # Reconstruction in the standardized space
    X_reconstructed_std = X_pca_reduced.dot(eigenvectors_reduced.T)
    print_matrix_info(X_reconstructed_std, "Reconstructed Standardized Data")
    
    # Inverse transform to get back to original scale
    X_reconstructed = scaler.inverse_transform(X_reconstructed_std)
    print_matrix_info(X_reconstructed, "Reconstructed Original Data")
    
    return X_reconstructed

def sample_from_pca_mixture_model(X_pca, y, eigenvectors, scaler, n_samples=30, n_components=2, n_clusters=3):
    """Generate samples using a Gaussian Mixture Model in PCA space."""
    print("\nStep 6: Sampling from a Gaussian Mixture Model in PCA space")
    
    # Fit a Gaussian Mixture Model to the PCA-transformed data
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X_pca[:, :n_components])
    
    # Print GMM parameters
    print("\nGaussian Mixture Model Parameters:")
    for i in range(n_clusters):
        print(f"Cluster {i+1}:")
        print(f"  Weight: {gmm.weights_[i]:.4f}")
        print(f"  Mean: {gmm.means_[i]}")
        print(f"  Covariance:\n{gmm.covariances_[i]}")
    
    # Sample from the GMM in PC space
    pc_samples, cluster_labels = gmm.sample(n_samples)
    print_matrix_info(pc_samples, "GMM Samples in PC Space")
    
    # Transform samples from PC space back to the original feature space
    sampled_points_std = pc_samples.dot(eigenvectors[:, :n_components].T)
    sampled_points = scaler.inverse_transform(sampled_points_std)
    
    # Map cluster labels to match original data colors
    # This is a heuristic approach to match colors - might need adjustment based on specific data
    unique_labels = np.unique(y)
    label_mapping = {}
    
    # Match GMM clusters to original data clusters based on cluster centers
    for gmm_cluster in range(n_clusters):
        gmm_center = gmm.means_[gmm_cluster]
        # Transform the GMM center back to original space - FIX: Reshape to 2D array
        transformed_center = gmm_center.dot(eigenvectors[:, :n_components].T).reshape(1, -1)
        gmm_center_original = scaler.inverse_transform(transformed_center)[0]  # Get the first row back
        
        # Calculate distances to original cluster centers
        distances = []
        for orig_cluster in unique_labels:
            cluster_center = np.mean(X[y == orig_cluster], axis=0)
            dist = np.linalg.norm(gmm_center_original - cluster_center)
            distances.append(dist)
        
        # Map to closest original cluster
        label_mapping[gmm_cluster] = unique_labels[np.argmin(distances)]
    
    # Apply mapping
    mapped_labels = np.array([label_mapping[label] for label in cluster_labels])
    
    return sampled_points, mapped_labels

# Generate sample data
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Run PCA step by step
X_pca, eigenvalues, eigenvectors, scaler, X_std = pca_step_by_step(X)

# Print final results
print("\nFinal Results:")
print("="*50)
print("Original Data Shape:", X.shape)
print("Transformed Data Shape:", X_pca.shape)
print("\nTotal Variance Explained:", np.sum(eigenvalues))
print("Individual Variance Explained:", eigenvalues)
print("\nPrincipal Components:")
for i, pc in enumerate(eigenvectors.T):
    print(f"PC{i+1}:", pc)

# Reconstruct original data points
X_reconstructed = reconstruct_data(X_pca, eigenvectors, scaler)

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(X - X_reconstructed))
print(f"\nReconstruction Error (MSE): {reconstruction_error:.6f}")

# Plot original vs reconstructed data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.5, label='Reconstructed')
plt.title('Reconstructed Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Demonstrate dimensionality reduction by using only the first principal component
X_reconstructed_1pc = reconstruct_data(X_pca, eigenvectors, scaler, n_components=1)
reconstruction_error_1pc = np.mean(np.square(X - X_reconstructed_1pc))
print(f"\nReconstruction Error with 1 PC (MSE): {reconstruction_error_1pc:.6f}")

# Sample from a Gaussian Mixture Model in PCA space
sampled_points_gmm, mapped_labels = sample_from_pca_mixture_model(X_pca, y, eigenvectors, scaler, n_samples=30, n_clusters=3)

# Plot original data vs GMM-based sampled data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(sampled_points_gmm[:, 0], sampled_points_gmm[:, 1], alpha=0.7, c=mapped_labels, cmap='viridis')
plt.title('Gaussian Mixture PCA Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

plt.tight_layout()
plt.show()