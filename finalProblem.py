# finalProblem.py
# Purpose: Objective function + simple guess-and-check minimization
# Name: Quinn Peters

import numpy as np


def fv(v):
    """
    Compute the objective function:
        f(v1, v2) = 2 + v1/40 + v2/30 + cos(v1*v2/20)
    where v is a list or array like [v1, v2].

    Returns
    -------
    f : float
        Objective value
    """
    v1 = float(v[0])
    v2 = float(v[1])
    f = 2.0 + v1 / 40.0 + v2 / 30.0 + np.cos((v1 * v2) / 20.0)

    print(f" f([ {v1:7.4f} , {v2:7.4f} ]) = {f:7.4f} ")
    return f


def clip_to_domain(v, lo=-10.0, hi=10.0):
    return np.clip(v, lo, hi)


def guess_and_check(v0, step0=2.0, iters=15):
    """
    A simple “smart” guess-and-check optimizer:

    - Keep a move direction d (2D vector).
    - If the last move improved f, keep moving in that direction (momentum).
    - If it got worse, reverse direction and shrink the step (backtracking).
    - Also tries coordinate nudges if stuck.

    This is basically a derivative-free pattern search.
    """
    vkm2 = clip_to_domain(np.array(v0, dtype=float))
    fkm2 = fv(vkm2)

    # Make an initial second guess by nudging v1
    d = np.array([1.0, 0.0])
    step = float(step0)

    vkm1 = clip_to_domain(vkm2 + step * d)
    fkm1 = fv(vkm1)

    # Choose a third guess based on whether that helped
    if fkm1 < fkm2:
        vk = clip_to_domain(vkm1 + step * d)   # keep going
    else:
        step *= 0.5
        d = -d
        vk = clip_to_domain(vkm2 + step * d)   # reverse + shrink

    fk = fv(vk)

    for _ in range(iters):
        # “logic” for v(k+1):
        # if f(k) < f(k-1): continue along last successful displacement
        # else: backtrack (reverse & shrink)
        last_disp = vk - vkm1

        if fk < fkm1:
            d = last_disp if np.linalg.norm(last_disp) > 1e-12 else d
            candidate = clip_to_domain(vk + d)  # same direction, same “size” as last move
        else:
            step *= 0.5
            d = -last_disp if np.linalg.norm(last_disp) > 1e-12 else -d
            candidate = clip_to_domain(vk + step * (d / (np.linalg.norm(d) + 1e-12)))

        fc = fv(candidate)

        # If we didn’t improve, try small coordinate moves 
        if fc >= fk:
            e1 = np.array([1.0, 0.0])
            e2 = np.array([0.0, 1.0])

            c1 = clip_to_domain(vk + step * e1)
            f1 = fv(c1)

            c2 = clip_to_domain(vk - step * e1)
            f2 = fv(c2)

            c3 = clip_to_domain(vk + step * e2)
            f3 = fv(c3)

            c4 = clip_to_domain(vk - step * e2)
            f4 = fv(c4)

            # pick best among candidate and coordinate nudges
            options = [(candidate, fc), (c1, f1), (c2, f2), (c3, f3), (c4, f4)]
            best_v, best_f = min(options, key=lambda t: t[1])
            candidate, fc = best_v, best_f

        # shift the history (k-2 <- k-1 <- k <- k+1)
        vkm2, fkm2 = vkm1, fkm1
        vkm1, fkm1 = vk, fk
        vk, fk = candidate, fc

    return vk, fk


if __name__ == "__main__":
    # Example run 
    v_best, f_best = guess_and_check([3.0, -4.0], step0=2.0, iters=10)
    v1, v2 = v_best[0], v_best[1]
    f = f_best
    print(f" f([ {v1:7.4f} , {v2:7.4f} ]) = {f:7.4f} ")
