import numpy as np
from scipy.stats import multivariate_normal


""" Wrapper Class for Better Presentataion"""
class GMMPredictionResults:
    def __init__(self, probabilities, component, component_scores):
        self.probabilities = probabilities
        self.component = component
        self.component_scores = component_scores


class GMMParameters:
    def __init__(self, means, covariances, weights, probabilities, log_likelihood):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.probabilities = probabilities
        self.log_likelihood = log_likelihood


# Main GMM model
class CustomGMM:
    def __init__(self, n_components, initializer=None, max_iter=200, covariance_type='full'):
        self.n_components = n_components
        self.initializer = initializer
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.tol = 1e-3

        # Model parameters
        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None
        self.log_likelihood = None
        self.n_iter = None
        self.converged = None


    def fit(self, X, initializer=None, verbose=False):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        n_samples, n_features = X.shape

        if self.initializer is not None:
            params = initializer(X, self.n_components)

            self.means = params['means']
            self.covariances = params['covariances']
            self.weights = params['weights']
            self.responsibilities = params['responsibilities']
        
        else:
            indices = np.linspace(0, n_samples - 1, self.n_components, dtype=int)

            self.means =  X[indices].copy()
            self.covariances =  np.array([np.eye(n_features) for _ in range(self.n_components)])
            self.weights =  np.ones(self.n_components) / self.n_components
            self.responsibilities =  np.zeros((n_samples, self.n_components))

        self.log_likelihood = -np.inf
        self.converged = False

        for iteration in range(self.max_iter):
            new_log_likelihood = self.expectation_step(X)

            if abs(new_log_likelihood - self.log_likelihood) < self.tol:
                self.converged = True
                break
            self.log_likelihood = new_log_likelihood

            self.maximization_step(X)
            if verbose:
                print(f"Iteration {iteration}, Log-likelihood: {self.log_likelihood:.4f}")

        self.n_iter = iteration
        print(f"Converged after {iteration} iterations" if self.converged else f"Reached max iterations ({self.max_iter})")
        print(f"Log-likelihood: {self.log_likelihood:.4f}")
        return self


    def expectation_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k], allow_singular=True)
            except:
                responsibilities[:, k] = self.weights[k] * 1e-10

        sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
        sum_responsibilities[sum_responsibilities == 0] = 1e-10
        responsibilities /= sum_responsibilities

        log_likelihood = np.sum(np.log(sum_responsibilities))
        self.responsibilities = responsibilities
        return log_likelihood


    def maximization_step(self, X):
        n_samples, n_features = X.shape
        N_k = self.responsibilities.sum(axis=0)

        # Weights update
        self.weights = N_k / n_samples

        # Means update
        new_means = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            if N_k[k] > 0:
                new_means[k] = (self.responsibilities[:, k:k+1] * X).sum(axis=0) / N_k[k]
        self.means = new_means

        # Covariances update
        new_covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            if N_k[k] > 0:
                diff = X - self.means[k]

                if self.covariance_type == 'full':
                    new_covariances[k] = np.dot((self.responsibilities[:, k:k+1] * diff).T, diff) / N_k[k]
                    new_covariances[k] += 1e-6 * np.eye(n_features)  # regularization

                elif self.covariance_type == 'diag':
                    cov_diag = np.sum(self.responsibilities[:, k:k+1] * (diff**2), axis=0) / N_k[k]
                    new_covariances[k] = np.diag(cov_diag + 1e-6)
        self.covariances = new_covariances


    def generate_samples(self, n_samples):
        component_indices = np.random.choice(
            self.n_components,
            size=n_samples,
            p=self.weights)

        samples = np.array([np.random.multivariate_normal(self.means[k], self.covariances[k]) for k in component_indices])
        return samples, component_indices

    
    def predict(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        probabilities = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            probabilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k], allow_singular=True)

        probabilities /= probabilities.sum(axis=1, keepdims=True)
        component_scores = np.max(probabilities, axis=1)
        component = np.argmax(probabilities, axis=1)
        
        return GMMPredictionResults(probabilities, component, component_scores)


    def get_params(self):
        return GMMParameters(self.means, self.covariances, self.weights, self.responsibilities, self.log_likelihood)