import numpy as np

def gerar_sistema(n, seed=None, intervalo=(1.0, 10.0)):
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.uniform(intervalo[0], intervalo[1], size=(n, n)).round(2)
    b = np.random.uniform(intervalo[0], intervalo[1], size=n).round(2)
    
    return A, b

from sympy import symbols, Eq

def gerar_equacoes(n, intervalo=(1.0, 10.0), seed=None):
    if seed is not None:
        np.random.seed(seed)

    variaveis = symbols(f'x:{n}') 
    
    A = np.random.uniform(intervalo[0], intervalo[1], size=(n, n)).round(2)
    b = np.random.uniform(intervalo[0], intervalo[1], size=n).round(2)

    for i in range(n):
        soma_outros = np.sum(np.abs(A[i])) - abs(A[i][i])
        A[i][i] = soma_outros + np.random.uniform(1.0, 5.0)  
        A[i][i] = round(A[i][i], 2)

    equacoes = []
    for i in range(n):
        expressao = sum(A[i][j] * variaveis[j] for j in range(n))
        equacao = Eq(expressao, b[i])
        equacoes.append(equacao)

    return equacoes, variaveis

def gerar_pontos(n, intervalo=(1.0, 10.0), seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    pontos = []
    xs_inter = []
    
    for i in range(n):
        x = np.random.uniform(intervalo[0], intervalo[1], size=1).round(2)
        y = np.random.uniform(intervalo[0], intervalo[1], size=1).round(2)
        x_inter = np.random.uniform(intervalo[0], intervalo[1], size=1).round(2)
        
        ponto = (float(x[0]), float(y[0]))
        pontos.append(ponto)
        xs_inter.append(float(x_inter[0]))

    return pontos, xs_inter

def gerar_pontos_baseado_funcao(n, funcao, intervalo=(0.0000, 1.0000), seed=None):
    x = symbols('x')
    
    if seed is not None:
        np.random.seed(seed)
    
    pontos = []

    for i in range(n):
        xi = np.random.uniform(intervalo[0], intervalo[1], size=1).round(2)
        yi = funcao.subs(x, float(xi))
        
        ponto = (float(xi), round(float(yi), 4))
        pontos.append(ponto)

    return pontos