from sympy import symbols, Mul, expand, Eq
from sympy.printing import sstr

def tabela_diferencas_divididas(xs, ys):
    n = len(xs)
    tabela = [ys.copy()]

    for i in range(1, n):
        linha_anterior = tabela[-1]
        nova_linha = []
        for j in range(n - i):
            numerador = linha_anterior[j+1] - linha_anterior[j]
            denominador = xs[j+i] - xs[j]
            if denominador == 0:
                denominador = 0.01  # evitar divisão por zero
            nova_linha.append(numerador / denominador)
        tabela.append(nova_linha)
    
    return [linha[0] for linha in tabela]  # retornamos apenas os coeficientes


def metodo_newton(xs, ys, xs_inter):
    x = symbols('x')
    coeficientes = tabela_diferencas_divididas(xs, ys)
    p_x = coeficientes[0]
    
    for i in range(1, len(coeficientes)):
        termo = 1
        for j in range(i):
            termo *= (x - xs[j])
        p_x += coeficientes[i] * termo
    
    pontos_inter = [p_x.subs(x, val) for val in xs_inter]
    return expand(p_x), pontos_inter


