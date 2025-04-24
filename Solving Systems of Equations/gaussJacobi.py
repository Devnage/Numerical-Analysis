# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 08:23:12 2025

@author: Acer
"""

import numpy as np
import pandas as pd

def main() -> None:
    A = np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1], [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]])
    b = np.array([0, 1, 0, 9, 8])
    x0 = np.array([0, 0, 0, 0, 0])
    error_tolerance = 1e-6
    max_iterations = 100
    
    jacobi(A, b, x0, error_tolerance, max_iterations)
    
def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, errorTol: float = 1e-6, maxIter: int = 100) -> None:
    table = []
    curr_x = x0
    
    D, L, U = splitting(A)
    D_inv = np.linalg.inv(D)
    Tj = D_inv @ (L + U)
    Cj = D_inv @ b
    
    if max(np.abs(np.linalg.eigvals(Tj))) >= 1:
        print("The method will not converge")
        return
    
    for i in range(maxIter + 1):
        new_x = (Tj @ curr_x) + Cj
        relJump = np.linalg.norm(new_x - curr_x, np.inf)/np.linalg.norm(curr_x, np.inf) if np.linalg.norm(curr_x) != 0 else np.inf
        table.append([i, curr_x[0], curr_x[1], curr_x[2], curr_x[3], curr_x[4], relJump])
        
        if  relJump < errorTol:
            break
        
        curr_x = new_x
        
    df = pd.DataFrame(table, columns = ["n", "x_1", "x_2", "x_3", "x_4", "x_5", "rel jump"])
    print(df.to_string(index = None))
    print()
    print(f"Initial vector: {x0}")
    print(f"Solution after {df.iloc[-1, 0]} iterations: {curr_x}")
    print(f"Asymptotic error constant upper bound: {np.linalg.norm(Tj, np.inf)}")
    print("Substituting the estimates in Ax = b:")
    print(A @ curr_x)

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