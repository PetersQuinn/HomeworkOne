# sinWork.py
# Purpose: Compute and plot sums of sinusoids (Fourier series partial sums) for n=5,10,15.
# Name: Quinn Peters

import numpy as np
import matplotlib.pyplot as plt


def fourier_square_partial(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute y(x; n) = (4/pi) * sum_{k=1..n} [ 1/(2k-1) * sin((2k-1)*x) ]
    without using an explicit sum() or inner for-loop.
    """
    k = np.arange(1, n + 1)                 # k = [1, 2, ..., n]
    m = 2 * k - 1                           # odd harmonics: 1, 3, 5, ...
    a = 1 / m                               # coefficients 1/(2k-1)

    # Make a column vector of coefficients (n x 1) and a row vector of x (1 x len(x))
    # sin(m^T * x) becomes an (n x len(x)) matrix of all sinusoids
    # Then (a^T) @ sin_matrix gives (1 x len(x)) which we flatten back to (len(x),)
    y = (4 / np.pi) * (a.reshape(1, -1) @ np.sin(m.reshape(-1, 1) * x.reshape(1, -1))).ravel()
    return y


def main():
    x = np.linspace(-np.pi, np.pi, 500)

    plt.figure(figsize=(9, 5))

    # single for-loop over the requested n values
    for n in (5, 10, 15):
        y = fourier_square_partial(x, n)
        plt.plot(x, y, label=f"n = {n}")

    plt.xlabel("x")
    plt.ylabel("y(x; n)")
    plt.title("Fourier series partial sums: sum of sinusoids")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
