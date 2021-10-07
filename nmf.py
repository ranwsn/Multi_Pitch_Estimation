from typing import Optional, Tuple

import numpy as np
from scipy import linalg

"""
A module for non-negative matrix factorization, trying to approximate a matrix
X using the factorization X = AW
"""


def mu_solver(X: np.array, dict_size: int, beta: float = 2,
              A: Optional[np.array] = None, W: Optional[np.array] = None,
              init_method: Optional[str] = None, delta_min: float = 1e-8,
              max_iterations: int = 100, verbose: bool = True,
              plot_flag: bool = True) -> dict:

    # preliminaries
    if init_method is None:
        init_method = "randn"
    n, m = X.shape
    history = {"d_beta": [], "d_euq": [], "delta_A": [], "delta_W": []}

    # initialize
    A, W = init_matrices(init_method, rows=n, cols=m, dict_size=dict_size,
                         X=X, A=A, W=W)

    # solve iteratively
    iteration = 1
    while iteration < max_iterations:

        # save last values
        A_old = A.copy()
        W_old = W.copy()

        # update A and W
        AW = A @ W
        A = A_old * ((((AW ** (beta-2)) * X) @ W.T) / ((AW ** (beta-1)) @ W.T))
        AW = A @ W
        W = W_old * ((A.T @ ((AW ** (beta-2)) * X))/(A.T @ (AW ** (beta-1))))

        # compute difference
        AW = A @ W
        d_beta = beta_divergence(X, AW, beta)
        d_euq = beta_divergence(X, AW, 2)
        delta_A = np.linalg.norm(A-A_old)
        delta_W = np.linalg.norm(W-W_old)
        history["d_beta"].append(d_beta)
        history["d_euq"].append(d_euq)
        history["delta_A"].append(delta_A)
        history["delta_W"].append(delta_W)

        # print
        if verbose:
            print(f"*** Iteration {iteration}: ***")
            print(f"d_beta: {d_beta}")
            print(f"d_euq: {d_euq}")
            print(f"delta_A: {delta_A}")
            print(f"delta_W: {delta_W}")

        # check stopping condition
        if (delta_A < delta_min) and (delta_W < delta_min):
            break



    if iteration == max_iterations:
        print("Warning! Algorithm may have not converged, as it reached"
              "maximal number of iterations")


def init_matrices(init_method: str, rows: int, cols: int, dict_size: int,
                  X: Optional[np.array] = None, A: Optional[np.array] = None,
                  W: Optional[np.array] = None) -> Tuple[np.array, np.array]:
    if init_method == "given":
        assert A is not None
        assert W is not None
    elif init_method == "randn":
        # TODO: add scaling according to X
        A = np.abs(np.random.randn(rows, dict_size))
        W = np.abs(np.random.randn(dict_size, cols))
    elif init_method == "uniform":
        # TODO: add scaling according to X
        A = np.random.rand(rows, dict_size)
        W = np.random.rand(dict_size, cols)
    elif init_method == "svd":
        assert dict_size <= min(rows, cols)
        U, s, Vh = linalg.svd(X)
        U = U[:, :dict_size]
        s = s[:dict_size]
        Vh = Vh[:dict_size, :]
        A = U * s
        W = s[:, None] * Vh
    else:
        raise NotImplementedError

    return A, W


def beta_divergence(x: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    """
    compute the beta divergence between 2 arbitrary ndarray objects
    :param x: first/left object
    :param y: second/right object
    :param beta: beta value, can be any number
    :return: d_beta(x | y), the beta divergence between x and y (not symmmetric)
    """

    assert x.shape == y.shape, "shapes do not match"

    b = beta
    if b == 0:  # itakura-saito divergence
        d = np.sum(x/y - np.log(x/y) - 1)
    elif b == 1:  # KL divergence
        d = np.sum(x * np.log(x / y) + (y-x))
    elif b == 2:  # euclidian distance
        d = 0.5 * np.linalg.norm(x-y)**2
    else:  # general case
        d = np.sum((x**b + (b-1)*(y**b) - b*x*(y**(b-1))) / (b*(b-1)))

    return d
