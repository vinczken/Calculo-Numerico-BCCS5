#Gauss Seidel
import numpy as np
from sympy import symbols, Eq, solve
from scipy.sparse.linalg import cg 

def metodo_gauss_seidel(equacoes, variaveis, x0 = 0, tol = 1e-6, iteracoes = 1000):
    n = len(variaveis)
    
    solucoes = [solve(equacoes[i], variaveis[i])[0] for i in range(n)]
    
    matrizA = np.zeros((n, n), dtype=float)
    matrizb = np.zeros(n, dtype=float)

    for i in range(0, n):
        lado_esq = equacoes[i].lhs
        matrizb[i] = float(equacoes[i].rhs)
        coefs = lado_esq.as_coefficients_dict()
        for j in range(n):
            matrizA[i][j] = float(coefs.get(variaveis[j], 0))
    
    x_aprox = np.linalg.solve(matrizA, matrizb)
    
    
    verificador = False
    for i in range(0, n):
        diagonal = abs(matrizA[i,i])
        soma_outros = np.sum(np.abs(matrizA[i])) - diagonal
        if diagonal > soma_outros:
            verificador = True
    
    if not verificador:
        return "Não é possível realizar este método (matriz não é diagonal dominante).", None, None
    
    valores = np.array([x0] * n, dtype=float)
    for iter in range(0, iteracoes):
        novos_valores = valores.copy()
        
        for i in range(0, n):
            contexto = {
                variaveis[j]: novos_valores[j] if j < i else valores[j] for j in range(n)
            }
            novos_valores[i] = float(solucoes[i].subs(contexto))
        
        if np.all(np.abs(novos_valores - x_aprox) <= tol):
            return novos_valores.tolist(), iter + 1, x_aprox

        valores = novos_valores
    
    return valores.tolist(), iteracoes, x_aprox