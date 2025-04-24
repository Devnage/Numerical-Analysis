import pandas as pd

f = lambda x: x**6 - x - 1
nMax = 100
errorTol = 1e-6
x0 = 1
x1 = 1.5

def secant(n, error, x0, x1):
    table = []
    for i in range(n):
        x_new = x1 - f(x1)*((x1 - x0)/(f(x1) - f(x0)))
        relJump = abs((x_new - x1)/(x1))
        table.append([i + 1, x_new, relJump])
        x0 = x1
        x1 = x_new
        if relJump < error:
            break
    return pd.DataFrame(table, columns = ["n", "x_n", "rel jump"])


df = secant(nMax, errorTol, x0, x1)
print(df.to_string(index = False))
print(df.iloc[-1, 1])