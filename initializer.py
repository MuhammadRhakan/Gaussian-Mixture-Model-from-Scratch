import numpy as np


def even_spaced_initializer(X, n_components):
    n_samples, n_features = X.shape
    indices = np.linspace(0, n_samples - 1, n_components, dtype=int)
    
    return {
        'means': X[indices].copy(),
        'covariances': np.array([np.eye(n_features) for _ in range(n_components)]),
        'weights': np.ones(n_components) / n_components,
        'responsibilities': np.zeros((n_samples, n_components))
    }


def kmeans_initializer(X, n_components, max_iter=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples, n_features = X.shape
    
    random_indices = np.choice(n_samples, n_components, replace=False)
    centroids = X[random_indices].copy()   
    cluster_labels = np.zeros(n_samples)
    
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        if np.all(cluster_labels == new_labels):
            break
        cluster_labels = new_labels

        new_centroids = np.array([
            X[cluster_labels == k].mean(axis=0) if np.any(cluster_labels == k) else centroids[k] 
            for k in range(n_components)
        ])
        centroids = new_centroids

    covariances = np.zeros((n_components, n_features, n_features))
    weights = np.zeros(n_components)
    responsibilities = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        cluster_points = X[cluster_labels == k]
        weights[k] = len(cluster_points) / n_samples
        
        if len(cluster_points) > 1:
            covariances[k] = np.cov(cluster_points, rowvar=False) + 1e-6 * np.eye(n_features)
        else:
            covariances[k] = np.eye(n_features)
            
        responsibilities[cluster_labels == k, k] = 1.0

    return {
        'means': centroids,
        'covariances': covariances,
        'weights': weights,
        'responsibilities': responsibilities
    }