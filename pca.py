import numpy as np


def pca(X, k):
    """
    Principle Component Analysis
    :param x: data
    :param k: number of components you want
    :return: k-dim data
    """
    # Compute mean of each feature
    mean = np.mean(X, axis=0)
    norm_X = X - mean
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)

    eig_vals, eig_vectors = np.linalg.eigh(scatter_matrix)

    # The magnitude of eigenvalue tells the strength of relationship
    s = np.where(eig_vals < 0)
    eig_vals[s] = eig_vals[s] * -1.0
    # Change the direction of eigenvectors
    eig_vectors[:, s] = eig_vectors[:, s] * -1.0

    # Sort eig_vec based on eig_val from highest to lowest
    # argsort: in ascending order, [::-1]: reverser the array order
    idx = np.argsort(eig_vals)[::-1]
    # Sort eigen vectors according to same indices
    eig_vectors = eig_vectors[:, idx]
    # Select the largest k eigenvectors
    principle_components = eig_vectors[:, :k]

    # New data of k-dim
    new_data = np.dot(norm_X, principle_components)

    return new_data, principle_components
