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
    m = 2 * k - 1                           # odd harmonics
    a = 1 / m                               # coefficients 
    y = (4 / np.pi) * (a.reshape(1, -1) @ np.sin(m.reshape(-1, 1) * x.reshape(1, -1))).ravel()
    return y


def main():
    x = np.linspace(-np.pi, np.pi, 500)

    plt.figure(figsize=(9, 5))

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
