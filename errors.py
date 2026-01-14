import numpy as np

# -------------------------
# Two MORE vector mistakes
# -------------------------

# y (vector mistake): using ^ expecting exponentiation (it's XOR for ints)
y = np.array([1, 2, 3]) ^ 2          # WRONG: bitwise XOR
y_fix = np.array([1, 2, 3]) ** 2     # RIGHT: exponentiation

# z (vector mistake): expecting .append to modify a NumPy array "in place"
z = np.array([5, 8, 13])
z.append(21)                          # WRONG: NumPy arrays have no append method
z_fix = np.append(z, 21)              # RIGHT: returns a new array


# -------------------------
# Two MORE matrix mistakes
# -------------------------

# E (matrix mistake): using * expecting matrix multiplication
E = np.array([[1, 2],
              [3, 4]])
F = np.array([[10],
              [20]])

E_wrong = E * F                       # WRONG: element-wise broadcasting (not matrix multiply)
E_fix = E @ F                         # RIGHT: matrix multiplication (2x2)@(2x1) -> (2x1)

# F (matrix mistake): trying to invert a non-square matrix
F = np.array([[1, 2, 3],
              [4, 5, 6]])             # 2x3 not square
F_wrong = np.linalg.inv(F)            # WRONG: inv requires a square matrix

# Fix: either use pseudoinverse, or solve a square system instead
F_fix = np.linalg.pinv(F)             # RIGHT: pseudoinverse exists for non-square matrices
