import pandas as pd

i = lambda l: (l/12)
j = lambda m: (1 + i(m))**36
f = lambda x: (675000*j(x)) + (12500*((j(x) - 1)/(i(x)))) - 1250000
nMax = 100
errorTol = 1e-6
x0 = 1

def dx(f, x):
    h = 1e-6
    return (f(x + h) - f(x - h)) / (2*h)

def newtonRaphson(n, error, x0):
    table = []
    for i in range(n):
        x_new = x0 - (f(x0) / dx(f, x0))
        relJump = abs((x_new - x0)/(x0))
        table.append([i + 1, x_new, relJump])
        x0 = x_new
        if relJump < error:
            break
    return pd.DataFrame(table, columns = ["n", "x_n", "rel jump"])

df = newtonRaphson(nMax, errorTol, x0)
print(df.to_string(index = False))
print(df.iloc[-1, 1])