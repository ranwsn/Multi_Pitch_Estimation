from typing import Optional, Tuple

import numpy as np
from scipy import linalg
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

"""
A module for non-negative matrix factorization, trying to approximate a matrix
X using the factorization X = AW
"""


def mu_solver(X: np.array, dict_size: int, beta: float = 2,
              A: Optional[np.array] = None, W: Optional[np.array] = None,
              init_method: Optional[str] = None, delta_min: float = 1e-8,
              max_iterations: int = 100, verbose: bool = True,
              plot_flag: bool = True) -> Tuple[np.array, np.array, dict]:
    # preliminaries
    if init_method is None:
        init_method = "randn"
    n, m = X.shape
    history = {"d_beta": [], "d_euc": [], "delta_A": [], "delta_W": [],
               "delta_A_rel": [], "delta_W_rel": []}

    # initialize
    A, W = init_matrices(init_method, rows=n, cols=m, dict_size=dict_size,
                         X=X, A=A, W=W)

    # solve iteratively
    iteration = 1
    while iteration <= max_iterations:

        # save last values
        A_old = A.copy()
        W_old = W.copy()

        # update A and W
        AW = A @ W
        A = A_old * ((((AW ** (beta - 2)) * X) @ W.T) / (
                    (AW ** (beta - 1)) @ W.T))
        AW = A @ W
        W = W_old * ((A.T @ ((AW ** (beta - 2)) * X)) / (
                    A.T @ (AW ** (beta - 1))))

        assert np.all(A >= 0), "numeric error - not all values are non-negative"
        assert np.all(W >= 0), "numeric error - not all values are non-negative"

        # compute difference
        AW = A @ W
        d_beta = beta_divergence(X, AW, beta)
        d_euc = beta_divergence(X, AW, 2)
        delta_A = np.linalg.norm(A - A_old)
        delta_W = np.linalg.norm(W - W_old)
        delta_A_rel = delta_A / np.linalg.norm(A_old)
        delta_W_rel = delta_W / np.linalg.norm(W_old)
        history["d_beta"].append(d_beta)
        history["d_euc"].append(d_euc)
        history["delta_A"].append(delta_A)
        history["delta_W"].append(delta_W)
        history["delta_A_rel"].append(delta_A_rel)
        history["delta_W_rel"].append(delta_W_rel)

        # check stopping condition
        if (delta_A_rel < delta_min) and (delta_W_rel < delta_min):
            break
        # TODO: consider stopping condition that depends on the cost

        # next iteration
        iteration += 1

    if iteration == max_iterations:
        print("Warning! Algorithm may have not converged, as it reached"
              "maximal number of iterations")

    # print
    if verbose:
        print(f"*** MU solver finished! ***")
        print(f"total iterations: {max_iterations}")
        print(f"d_beta: {d_beta}")
        print(f"d_euc: {d_euc}")
        print(f"delta_A_rel: {delta_A_rel}")
        print(f"delta_W_rel: {delta_W_rel}")

    # plot
    if plot_flag:
        iter_vec = np.arange(iteration)

        # plot d_beta and d_euc as function of iteration
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("d_beta", "d_euclidean"))
        fig.add_trace(
            go.Scatter(x=iter_vec, y=history["d_beta"]), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=iter_vec, y=history["d_euc"]), row=2, col=1
        )

        fig.update_layout(showlegend=False)
        fig.show()

        # plot delta as function of iteration
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                            subplot_titles=("delta_A", "delta_W",
                                            "delta_A_rel [dB]",
                                            "delta_W_rel [dB]"))
        fig.add_trace(
            go.Scatter(x=iter_vec, y=history["delta_A"]), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=iter_vec, y=history["delta_W"]), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=iter_vec, y=20*np.log10(history["delta_A_rel"])), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=iter_vec, y=20*np.log10(history["delta_W_rel"])), row=2, col=2
        )

        fig.update_layout(showlegend=False)
        fig.show()


        # plot output matrices
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("A (dictionary)", "W (weights)"))
        fig.add_trace(
            go.Heatmap(z=A, coloraxis='coloraxis'), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=W, coloraxis='coloraxis2'), row=1, col=2
        )
        fig.update_layout(coloraxis=dict(colorbar_x=0.46),
                          coloraxis2=dict(colorbar_x=1.0075))
        fig.show()

        # plot reconstructed matrix
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("X (org)", "X (reconstructed)"))
        fig.add_trace(
            go.Heatmap(z=X, coloraxis="coloraxis"), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=AW, coloraxis="coloraxis"), row=1, col=2
        )
        fig.show()

        pass

    return A, W, history


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
        d = np.sum(x / y - np.log(x / y) - 1)
    elif b == 1:  # KL divergence
        d = np.sum(x * np.log(x / y) + (y - x))
    elif b == 2:  # euclidian distance
        d = 0.5 * np.linalg.norm(x - y) ** 2
    else:  # general case
        d = np.sum((x ** b + (b - 1) * (y ** b) - b * x * (y ** (b - 1))) / (
                    b * (b - 1)))

    return d


# X = np.random.randn(200,400)
A_org = np.abs(np.random.randn(200, 100))
W_org = np.abs(np.random.randn(100, 400))
X = A_org @ W_org
A,W,history = mu_solver(X, dict_size=20, beta=1, init_method="randn", max_iterations=2000)
pass
