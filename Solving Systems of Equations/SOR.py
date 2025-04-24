import numpy as np
import pandas as pd

def main() -> None:
    A = np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1], [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]])
    b = np.array([0, 1, 0, 9, 8])
    x0 = np.array([0, 0, 0, 0, 0])
    omega = 1.05
    error_tolerance = 1e-6
    max_iterations = 100
    
    SOR(A, b, x0, omega, error_tolerance, max_iterations)

def SOR(A: np.ndarray, b: np.ndarray, x0: np.ndarray, omega: float, errorTol: float = 1e-6, maxIter: int = 100) -> None:
    table = []
    curr_x = x0

    D, L, U = splitting(A)
    M = ((1/omega) * D) - L
    M_inv = np.linalg.inv(M)
    N = ((1/omega - 1) * D) + U
    T_omega = M_inv @ N
    C_omega = M_inv @ b
    
    print(max(np.abs(np.linalg.eigvals(T_omega))))
    print(-np.log(max(np.abs(np.linalg.eigvals(T_omega)))))
    if max(np.abs(np.linalg.eigvals(T_omega))) >= 1:
        print("The method will not converge.")
        return
    
    for i in range(maxIter + 1):
        new_x = (T_omega @ curr_x) + C_omega
        relJump = np.linalg.norm(new_x - curr_x, np.inf)/np.linalg.norm(curr_x, np.inf) if np.linalg.norm(curr_x) != 0 else np.inf
        table.append([i, curr_x[0], curr_x[1], curr_x[2], curr_x[3], curr_x[4], relJump])
        
        if  relJump < errorTol:
            break
        
        curr_x = new_x
        
    df = pd.DataFrame(table, columns = ["n", "x_1", "x_2", "x_3", "x_4", "x_5", "rel jump"])
    print(df.to_string(index = None))
    print()
    print(f"Solution found after {df.iloc[-1, 0]} iterations.")
    print()
    print(f"x0 = {x0}^T")
    print(f"x* = {curr_x}^T")
    print(f"Asymptotic error constant upper bound: {np.linalg.norm(T_omega, np.inf)}")
    print(f"Ax* = {A @ curr_x}^T:")
    print()
    print()
    
def splitting(A: np.ndarray) -> np.ndarray:
    n = np.shape(A)
    D, L, U = [np.zeros_like(A) for _ in range(3)]
    
    for i in range(n[0]):
        for k in range(n[0]):
            if i < k:
                U[i][k] = -A[i][k]
            elif i > k:
                L[i][k] = -A[i][k]
            else:
                D[i][k] = A[i][k]
    return D, L, U

if __name__ == "__main__":
    main()

# Code for number 2
# import matplotlib.pyplot as plt
# from scipy.optimize import brentq

# def f(omega):
#     A = np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1], [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]])
#     D, L, U = splitting(A)
#     M = ((1/omega) * D) - L
#     N = ((1/omega - 1) * D) + U
#     T_omega = np.linalg.inv(M) @ N
#     return np.max(np.abs(np.linalg.eigvals(T_omega))) - 1

# # print(f(0.000001))
# # omega_vals = np.linspace(-2, 2, 100)
# # y_vals = np.vectorize(f)(omega_vals)
# # # plt.plot(omega_vals, y_vals)
# # # plt.show()

# root = brentq(lambda omega: f(omega), 1, 2)
# print(f"Root: {root}")
# print(f"f({root}) = {f(root)}")

# omega_vals = np.linspace(1e-8, root, 100)
# y_vals = np.vectorize(f)(omega_vals) + 1
# plt.plot(omega_vals, y_vals)
# plt.show()
