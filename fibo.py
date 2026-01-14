# fibo.py
# Purpose: Generate Fibonacci sequences and visualize convergence to the golden ratio value.
# Name: Quinn Peters

import numpy as np
import matplotlib.pyplot as plt


def seqnce(N):
    """
    Returns a Fibonacci sequence of length N+1 (F_0 through F_N) for N > 1.

    Returns
    -------
    j : np.ndarray
        Indices from 0 to N (inclusive).
    F : np.ndarray
        Fibonacci numbers F_0 ... F_N with F_0 = 1, F_1 = 1.
    """
    if N <= 1:
        raise ValueError("N must be greater than 1.")

    # (a) Compute Fibonacci sequence F_0 ... F_N using only F_0=1 and F_1=1
    F = np.zeros(N + 1, dtype=np.int64)
    F[0] = 1
    F[1] = 1
    for k in range(2, N + 1):
        F[k] = F[k - 1] + F[k - 2]

    # (b) Indices j = [0 ... N] using np.linspace
    j = np.linspace(0, N, N + 1).astype(int)

    return j, F


def plot(N):
    """
    Plots:
      1) Fibonacci sequence F_j vs j
      2) Ratio sequence r_j = F_j / F_{j+1} vs j, showing convergence to phi

    where phi solves 1/phi = 1 + phi, i.e. phi = (sqrt(5) - 1)/2 ≈ 0.618.
    """
    # (a) Get j and F from seqnce(N)
    j, F = seqnce(N)

    # (b) Ratio sequence (golden ratio values) from F
    r = F[:-1] / F[1:]
    j_r = j[:-1]

    # Golden ratio value as defined in the prompt: 1/phi = 1 + phi
    phi = (np.sqrt(5) - 1) / 2

    # (c) Plot both sequences with labeled axes
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(j, F, marker="o")
    axes[0].set_ylabel("Fibonacci number $F_j$")
    axes[0].set_title("Fibonacci sequence")

    axes[1].plot(j_r, r, marker="o", label=r"$F_j/F_{j+1}$")
    axes[1].axhline(phi, linestyle="--", label=rf"$\varphi = (\sqrt{{5}}-1)/2 \approx {phi:.6f}$")
    axes[1].set_xlabel("Index $j$")
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Convergence of consecutive ratios")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot(20)
