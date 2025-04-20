import numpy as np
from sympy import symbols, Mul, expand, Eq
from sympy.printing import sstr
import copy

def diferencas_divididas(xs, ys, f_x):
    n = len(f_x)
    if 3 > n:
        if n == 1:
            index = xs.index(f_x[0])
            return ys[index]
        if n == 2:
            x0 = xs.index(f_x[0])
            x1 = xs.index(f_x[1])
            divisor = (x1 - x0)
            if divisor == 0:
                divisor = 0.01
            return ((ys[x1] - ys[x0])/divisor)
    else:
        diferenca = diferencas_divididas(xs, ys, f_x[1:])
        dividida = diferencas_divididas(xs, ys, f_x[:-1])
        divisor = (f_x[n - 1] - f_x[0])
        if divisor == 0:
            divisor = 0.01
        return (diferenca - dividida) / divisor

def metodo_newton(xs, ys, xs_inter):
    x = symbols('x')
    p_x = 0
    
    fs = []
    for i in range(1, len(xs) + 1):
        valores = xs[0:i]
        f_atual = diferencas_divididas(xs, ys, valores)
        fs.append(f_atual)
    
    p_x += fs[0]
    
    i = 1
    for j in range(1, len(fs)):
        f_x = fs[i]
        equacao = 1
        for k in range(0, j):
            expressao = (x - xs[k])
            equacao *= expressao
        p_x += f_x * equacao
        i += 1
    
    pontos_inter = [ p_x.subs(x, xs_inter[i]) for i in range(len(xs_inter)) ]
    return expand(p_x), pontos_inter
