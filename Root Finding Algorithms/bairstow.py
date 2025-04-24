from sympy import *
import pandas as pd
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt

x = Symbol("x")

def chebyshev(n):
    if n == 0:
        poly = 1
    elif n == 1:
        poly = x
    elif n >= 2:
        poly = (2*x*(chebyshev(n - 1)) - chebyshev(n - 2))
    return Poly(poly, x).as_expr()

function = chebyshev(10)
errorTol = 1e-6
max_iter = 100
u = -1
v = 1

xval = np.linspace(-1, 1, 100)
yval = lambdify(x, function)(xval)
plt.plot(xval, yval)
plt.show()

def quadraticRoots(a, b, c):
    d = (b**2) - (4*a*c)
    r1 = (-b + cm.sqrt(d)) / (2*a)
    r2 = (-b - cm.sqrt(d)) / (2*a)
    
    return [r1 , r2]

def bairstow(func, u, v, Nmax, epsilon):
    table = []
    
    U = np.array([u, v], dtype = complex)
    
    a = func.as_poly().all_coeffs()[::-1]
    n = len(a)
    b = np.zeros(n, dtype = complex)
    f = np.zeros(n, dtype = complex)
    
    for i in range(Nmax):
        for k in range(n - 3, -1, -1):
            b[k] = (a[k + 2]) - (U[0]*b[k + 1]) - (U[1]*b[k + 2])
            f[k] = (b[k + 2]) - (U[0]*f[k + 1]) - (U[1]*f[k + 2])
        
        c = (a[1]) - (U[0]*b[0]) - (U[1]*b[1])
        g = (b[1]) - (U[0]*f[0]) - (U[1]*f[1])
        d = (a[0]) - (U[1]*b[0])
        h = (b[0]) - (U[1]*f[0])
        
        term1 = 1 / ((U[1]*(g**2)) + (h*(h - (U[0]*g))))
        term2 = np.array([[-h, g], [(-g*U[1]), ((g*U[0]) - h)]], dtype = complex)
        term3 = np.array([c, d], dtype = complex)
        
        Unew = U - (term1*np.matmul(term2, term3))
        relJump = abs(U - Unew) / abs(U)
        U = Unew
        
        table.append([i + 1, U[0], relJump[0], U[1], relJump[1]])
        
        if max(relJump) < epsilon:
            break
        
    print(pd.DataFrame(table, columns = ["n", "u_n", "rel jump (u)", "v_n", "rel jump (v)"]).to_string(index = None))
    print()
    qf = Poly([1, U[0], U[1]], x)
    roots = quadraticRoots(*qf.all_coeffs())
    print(f"Approximate quadratic factor: {qf.as_expr()}")
    print(f"Roots = {roots}")
    print(f"f({roots[0]}) = {lambdify(x, function)(roots[0])}")
    print(f"f({roots[1]}) = {lambdify(x, function)(roots[1])}")
    print()
    
    return roots, div(func, qf)[0]

def deflate(func, u, v, Nmax, epsilon):
    roots = []
    curr = func
    for i in range(degree(func, x), 0, -2):
        print(f"Bairstow's Method (Degree {i}): {curr.as_expr()}")
        if degree(curr, x) == 2:
            r = quadraticRoots(*curr.all_coeffs())
            print(f"Roots from Quadratic Equation: {r}")
            print()
            roots = roots + r
            break
        r, deflated = bairstow(curr, u, v, Nmax, epsilon)
        curr = deflated
        roots = roots + r
    print(f"Therefore, the roots of the original function are:")
    for i in range(len(roots)):
        print(f"x{i + 1} = {roots[i]}, f({roots[i]}) = {lambdify(x, function)(roots[i])}")

deflate(function, u, v, max_iter, errorTol)