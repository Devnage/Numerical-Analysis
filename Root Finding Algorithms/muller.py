from sympy import *
import cmath as cm
import pandas as pd

x0 = -3
x1 = 1
x2 = 2

x = Symbol("x")
Q = 2 + 0 + 2 + 2 + 0 + 1 + 0 + 9 + 8
f = x**4 + 12*(x**3) - 25*(x**2) + x - Q

Nmax = 1000
errorTol = 1e-6

def muller(fsym, triple, n, epsilon):
    x0, x1, x2 = triple
    table = []
    f = lambdify(x, fsym)
    for i in range(n):
        f0, f1, f2 = [f(x) for x in [x0, x1, x2]]
        a = ((f0) / ((x0 - x1)*(x0 - x2))) + ((f1)/((x1 - x0)*(x1 - x2))) + ((f2)/((x2 - x0)*(x2 - x1)))
        b = ((f0*(x2**2 - x1**2)) + (f1*(x0 - x2)*(x0 + x2)) + (f2*(x1**2 - x0**2)))/((x0 - x1)*(x0 - x2)*(x1 - x2))
        c = ((f0*x1*x2)/((x0 - x1)*(x0 - x2))) + ((f1*x0*x2)/((x1 - x0)*(x1 - x2))) + ((f2*x0*x1)/((x2 - x0)*(x2 - x1)))  

        rootp = (2*c) / (-b - cm.sqrt((b**2) - (4*a*c)))
        rootm = (2*c) / (-b + cm.sqrt((b**2) - (4*a*c)))
        
        x0 = x1
        x1 = x2
        x2 = rootp if abs(rootp - x2) < abs(rootm - x2) else rootm
        relJump = abs((x2 - x1)/x2)
        table.append([i + 1, x2, relJump])

        if relJump < epsilon:
            break
    
    df = pd.DataFrame(table, columns = ["n", "x_n", "relJump"])
    iterations = df.iloc[-1, 0]
    root = df.iloc[-1, 1]

    print(df.to_string(index = False))
    print(f"Root: {root} ({iterations} iterations)")
    print(f"f{root}: {f(root)} ")
    print()
    
    q = div(fsym, (x - root))[0]
    return q, degree(q, x), root

def deflate(fsym, triple, n, epsilon):
    roots = []
    fdegree = degree(fsym, x)
    orig = fdegree

    for i in range(fdegree, 0, -1):
        print(f"Muller's Method (Iteration {orig - i + 1}), Degree {fdegree}: {fsym}")
        if fdegree > 1:
            fsym, fdegree, r = muller(fsym, triple, n, epsilon)
        else:
            b = fsym.coeff(x, 0)
            a = fsym.coeff(x, 1)
            r = -b/a
            print(f"Root: x = -b/a = {r}")
            print(f"f({r}) = {a*r + b}")
        roots.append(r)
    
    print()
    print("Therefore, the roots are:")
    for r in roots:
        print(complex(r))

deflate(f, [x0, x1, x2], Nmax, errorTol)