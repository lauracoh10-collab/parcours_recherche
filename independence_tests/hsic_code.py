"""
Small HSIC helper utilities for independence testing.
This file is a lightweight placeholder with a simple RBF kernel and an empirical HSIC estimator.
"""
import numpy as np


def rbf_kernel(x, sigma=1.0):
    """Compute RBF kernel matrix for 1D or 2D inputs.

    x: array-like shape (n,) or (n, d)
    sigma: bandwidth
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    sq_dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    return np.exp(-sq_dists / (2 * sigma ** 2))


def center_matrix(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic_empirical(x, y, sigma_x=1.0, sigma_y=1.0):
    """Compute empirical HSIC statistic (biased estimator).

    x, y: 1D or 2D arrays with same number of rows
    Returns: scalar HSIC value
    """
    K = rbf_kernel(x, sigma=sigma_x)
    L = rbf_kernel(y, sigma=sigma_y)
    n = K.shape[0]
    Kc = center_matrix(K)
    Lc = center_matrix(L)
    return np.trace(Kc @ Lc) / (n - 1) ** 2


if __name__ == "__main__":
    # tiny smoke test
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    print("HSIC (independent gaussians):", hsic_empirical(x, y))
