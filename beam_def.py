# beam_def.py
# Purpose: Find where a point load on a simply supported beam maximizes the right-end rotation.
# Name: Quinn Peters

import numpy as np
import matplotlib.pyplot as plt


def main():
    # 100 values of r = a/L in [0, 1]
    r = np.linspace(0, 1, 100)

    # Dimensionless end rotation:
    # theta2 = F*a*b*(L+a)/(6*L*E*I), b = L-a
    # (E*I*theta2)/(F*L^2) = (a*b*(L+a)) / (6*L^3) = (r*(1-r)*(1+r))/6
    dim = (r * (1 - r) * (1 + r)) / 6  # one line, element-wise ops only

    i_max = np.argmax(dim)
    r_max = r[i_max]
    dim_max = dim[i_max]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(r, dim, label=r"$(EI\theta_2)/(F L^2)$")
    plt.plot(r_max, dim_max, "ro", label="max")

    plt.xlabel(r"$a/L$")
    plt.ylabel(r"$(EI\theta_2)/(F L^2)$")
    plt.title("Dimensionless right-end rotation vs load location")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Max at a/L = {r_max:.6f}")
    print(f"Max (EI*theta2)/(F*L^2) = {dim_max:.6f}")


if __name__ == "__main__":
    main()
