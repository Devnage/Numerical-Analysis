import pandas as pd

a0 = 1
b0 = 2

Nmax = 1000
epsilon = 10 ** -3

def f(x):
  return x**6  - x - 1

def bisection(f, a: float, b: float, errorTol: float = 1e-6, Nmax: int = 100) -> pd.DataFrame:
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

def main() -> None:
  if f(a0) * f(b0) < 0:
    df1 = bisection(f, a0, b0, epsilon, Nmax)
    print(df1.to_string(index = False))
    print()
    print(f"Root found at {df1.iloc[-1, 3]}, with function value {df1.iloc[-1, 5]}")
    print()
  else:
    print("Initial conditions are not appropriate.")

if __name__ == "__main__":
   main()