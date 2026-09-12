import numpy as np
import pandas as pd

def ledoit_wolf_shrinkage(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes the Ledoit-Wolf (2004) shrinkage covariance matrix.
    Target F: Diagonal matrix with equal average sample variance.
    
    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Return series of N assets over T time periods.
        
    Returns
    -------
    sigma_shrunk : np.ndarray, shape (N, N)
        The optimal regularized covariance matrix.
    delta_star : float
        Optimal shrinkage intensity between 0 and 1.
    """
    T, N = X.shape
    if T < 2:
        raise ValueError("Sample size T must be at least 2.")

    # 1. Demean the returns
    X_mean = np.mean(X, axis=0)
    Y = X - X_mean  # Shape: (T, N)

    # 2. Sample covariance matrix S (unbiased: 1 / (T - 1))
    S = (Y.T @ Y) / (T - 1)

    # 3. Shrinkage Target F = mu * I
    # mu is the mean of the sample variances (diagonal elements of S)
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # 4. Frobenius distance between S and F: gamma^2 = ||S - F||_F^2
    gamma_sq = np.sum((S - F) ** 2)

    # If S is already identical to F, no shrinkage needed
    if gamma_sq == 0:
        return S, 0.0

    # 5. Sum of asymptotic variances of entries of S: pi_hat
    # Vectorized element-wise outer product deviation:
    # Z_t = (y_ti * y_tj - S_ij)
    # pi_hat = sum_ij [ (1/T) * sum_t (y_ti * y_tj - S_ij)^2 ]
    # Using tensor contraction (einsum) to avoid slow loops:
    Y_cross = Y[:, :, None] * Y[:, None, :]  # Shape: (T, N, N)
    S_deviations = Y_cross - S[None, :, :]
    
    # Asymptotic variance term (unbiased scaling 1 / T)
    pi_mat = np.sum(S_deviations ** 2, axis=0) / T
    pi_hat = np.sum(pi_mat)

    # 6. Shrinkage intensity: kappa = (pi_hat - rho_hat) / gamma^2
    # For target F = mu * I, rho_hat captures covariance between S and F.
    # Analytical derivation for F = mu * I sets rho_hat to:
    # rho_hat = sum_i pi_ii / N (projected onto the diagonal)
    rho_hat = np.sum(np.diag(pi_mat))

    # 7. Optimal shrinkage intensity delta*
    kappa = (pi_hat - rho_hat) / gamma_sq
    delta_star = np.clip(kappa / T, 0.0, 1.0)

    # 8. Shrunk Covariance Matrix
    sigma_shrunk = (1.0 - delta_star) * S + delta_star * F

    return sigma_shrunk, float(delta_star)
