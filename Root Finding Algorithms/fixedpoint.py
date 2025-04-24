import pandas as pd
import numpy as np

Nmax = 100
errorTol = 1e-6
x0 = 1

def arccot(x):
    return np.pi/2 - np.arctan(x)

def cot(x):
    return -np.tan(x + np.pi/2)

def deriv(f, x):
    h = 1e-6
    return (f(x + h) - f(x - h)) / (2*h)

N = 1098
Z = 4116
f = lambda x: arccot(x) + (N/Z)
g = lambda x: cot(x - (N/Z))

def fixedPoint(f, x0, n, epsilon):
    table = []

    for i in range(n):
        x_new = f(x0)
        relJump = abs((x_new - x0)/(x0))
        x0 = x_new
        table.append([i + 1, x_new, relJump, deriv(f, x_new)])
        if relJump < epsilon:
            break

    lhs = lambda x: x - arccot(x)
    rhs = N/Z
    df = pd.DataFrame(table, columns = ["n", "x_n", "relJump", "asympConstant"])
    iterations = df.iloc[-1, 0]
    solution = df.iloc[-1, 1]
    estimate = lhs(solution)

    print(df.to_string(index = False))
    print()
    if iterations != n:
        print(f"Solution: {solution} ({iterations} iterations)")
        print(f"Estimated value: {estimate} ({abs(estimate - rhs)/ rhs} relative error)")
        print()
    else:
        print("Did not converge to a solution.")
        print()

fixedPoint(f, x0, Nmax, errorTol)
fixedPoint(g, x0, Nmax, errorTol)