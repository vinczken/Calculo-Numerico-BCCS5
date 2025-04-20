#Eliminação de Gauss

import numpy as np
def eliminacao_gauss_sem_pivoteamento(A, b):
    tamanho = len(A)
    
    Ab = np.concatenate((A, b.reshape(tamanho, 1)), axis=1)
    
    for coluna in range(0, tamanho): 
        for linha in range(0, tamanho):
            if Ab[linha, coluna] > Ab[coluna, coluna] and Ab[linha, coluna] != 0:
                Ab[[coluna, linha]] = Ab[[linha, coluna]]

    for coluna in range(0, tamanho):
        for linha in range(coluna + 1, tamanho):
            fator = Ab[linha, coluna] / Ab[coluna, coluna]
            Ab[linha, coluna:] = Ab[linha, coluna:] - fator * Ab[coluna, coluna:] # Operação nas linhas

    if Ab[tamanho - 1, tamanho - 1] == 0:
        return None

    x = np.zeros(tamanho)
    x[tamanho - 1] = Ab[tamanho - 1, tamanho] / Ab[tamanho - 1, tamanho - 1]
    for i in range(tamanho - 2, -1, -1):
        soma = 0
        for j in range(i + 1, tamanho):
            soma += Ab[i, j] * x[j]
        x[i] = round((Ab[i, tamanho] - soma) / Ab[i, i],4)
    return x
