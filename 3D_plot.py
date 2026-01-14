# 3D_plot.py
# Purpose: Create a 3D surface plot and a contour plot of the saddle function
#f(x, y) = 2x^2 - 3xy - 4y^2 on -10<=x<=10, -10<=y<=10
# Name: Quinn Peters

import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return 2 * x**2 - 3 * x * y - 4 * y**2


def main():
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # -------- 3D surface plot --------
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title(r"Surface plot of $f(x,y)=2x^2-3xy-4y^2$")

    plt.tight_layout()
    plt.show()

    # -------- Contour plot --------
    plt.figure(figsize=(8, 6))
    cs = plt.contour(X, Y, Z, levels=25)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Contour plot of $f(x,y)=2x^2-3xy-4y^2$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
