import numpy as np


class KNN:
    def __init__(self, K: int = 3, window_size: str = 'fixed') -> None:
        self.K = K
        self.window_size = window_size  # 'fixed' or 'variable'
        self.X_train = np.array([])
        self.y_train = np.array([])

    # START DISTANCE
    def minkowski(self, x: np.array, p: int) -> np.array:
        return np.power(np.sum(np.power(np.abs(self.X_train - x), p), axis=1), 1 / p)

    def cosine_similarity(self, x: np.array) -> np.array:
        dot = np.sum(self.X_train * x, axis=1)
        return 1 - dot / (np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x))

    def manhattan(self, x: np.array) -> np.array:
        return np.sum(np.abs(self.X_train - x), axis=1)

    def distances(self, X: np.array, metric: str) -> np.array:
        if metric == 'minkowski':
            return np.apply_along_axis(lambda x: self.minkowski(x, p=2), 1, X)
        elif metric == 'cosine':
            return np.apply_along_axis(self.cosine_similarity, 1, X)
        elif metric == 'manhattan':
            return np.apply_along_axis(self.manhattan, 1, X)

    # END DISTANCE
    # START KERNEL

    def uniform_kernel(self, x: np.array) -> np.array:
        return np.ones(len(x))

    def gaussian_kernel(self, x: np.array) -> np.array:
        return np.exp(-0.5 * (x ** 2))

    def triangular_kernel(self, x: np.array) -> np.array:
        return np.maximum(0, 1 - np.abs(x))

    def epanechnikov_kernel(self, x: np.array) -> np.array:
        return np.maximum(0, 3 / 4 * (1 - x ** 2))

    # END KERNEL

    def fit(self, X: np.array, y: np.array) -> None:
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)

    def generate_predictions(self, idx_neighbours: np.array, kernel_func, weights=None) -> np.array:
        if weights is None:
            weights = kernel_func(idx_neighbours[:, -1])
        y_pred = np.average(self.y_train[idx_neighbours], weights=weights, axis=1)
        return y_pred

    def predict(self, X: np.array, metric: str, kernel: str, weights: np.array = None,
                window_size: int = -1) -> np.array:
        if self.window_size == 'fixed' or window_size < 0:
            idx_neighbours = self.distances(X, metric).argsort()[:, :self.K]
        else:
            idx_neighbours = self.distances(X, metric).argsort()[:, :window_size]

        kernel_func = self.uniform_kernel
        if kernel == 'gaussian':
            kernel_func = self.gaussian_kernel
        elif kernel == 'triangular':
            kernel_func = self.triangular_kernel
        elif kernel == 'epanechnikov':
            kernel_func = self.epanechnikov_kernel

        y_pred = self.generate_predictions(idx_neighbours, kernel_func, weights)
        return y_pred