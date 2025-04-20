import numpy as np
from sympy import symbols, Mul, expand, Eq
from sympy.printing import sstr
import copy

def metodo_lagrange(pontos, xs_inter):
    n = len(pontos)
    
    x, y = symbols('x y') 
    lagrange = []
    
    for i in range(0, n):
        ponto = pontos[i]
        outros = copy.deepcopy(pontos)
        outros = list(outros)
        outros.remove(ponto)
        fatores = []
        denominador = 1
        for outro in outros:
            x = symbols('x')
            x0 = ponto[0]
            xn = outro[0]
            fatores.append(x - xn)
            denominador *= (x0 - xn)
        
        numerador = Mul(*fatores, evaluate=False)
        if denominador == 0:
            denominador = 0.0001
        expressao = numerador / denominador
        lagrange.append(expressao)
    
    #print(lagrange)
    
    p_x = 0
    for i in range(0, n):
        yn = pontos[i][1]
        lagn = lagrange[i]
        expressao = yn * lagn
        p_x += expressao
    
    pontos_inter = [ p_x.subs(x, xs_inter[i]) for i in range(len(xs_inter)) ]
    
    return expand(p_x), pontos_inter

