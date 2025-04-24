import pandas as pd
import numpy as np

i = lambda l: (l/12)
j = lambda m: (1 + i(m))**36
f = lambda x: (675000*j(x)) + (12500*((j(x) - 1)/(i(x)))) - 1250000
n = 100
tol = 1e-6

def bisection(f, a, b, errorTol, Nmax):
    table = []
    for i in range(Nmax):
        c = (a + b) * 0.5  
        fc = f(c)
        table.append([i, a, b, c, b - a, fc])
        if (b - a) < errorTol or fc == 0:
            break
        if fc*f(a) < 0:
            b = c
        else:
            a = c
    return pd.DataFrame(table, columns = ["Iteration", "a", "b", "c", "b - a", "f(c)"]) 

def regulaFalsi(f, a, b, errorTol, Nmax):
    table = []
    for i in range(Nmax):
        fa = f(a)
        c = np.dot([a, -b], [f(b), fa]) / (f(b) - fa)
        fc = f(c)
        table.append([i, a, b, c, fc])
        if abs(fc) < errorTol or fc == 0:
            break
        if fc*fa < 0:
            b = c
        else:
            a = c
    return pd.DataFrame(table, columns = ["Iteration", "a", "b", "c", "f(c)"])


df1 = bisection(f, 0.04, 0.05, tol, n)
print(df1.to_string(index = False))
print()
print(f"Root found at {df1.iloc[-1, 3]}, with function value {df1.iloc[-1, 5]}")
print()
df2 = regulaFalsi(f, 0.04, 0.05, df1.iloc[-1, 4], n)
print(df2.to_string(index = False))
print()
print(f"Root found at {df2.iloc[-1, 3]}, with function value {df2.iloc[-1, 4]}")
