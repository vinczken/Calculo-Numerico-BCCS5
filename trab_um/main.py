import numpy as np
from sympy import symbols, sin, cos
from eliminacao_gauss_sem_pivoteamento import eliminacao_gauss_sem_pivoteamento
from gerar_valores import gerar_sistema, gerar_equacoes, gerar_pontos, gerar_pontos_baseado_funcao
from gauss_seidel import metodo_gauss_seidel
from jacobi import metodo_jacobi
from lagrange import metodo_lagrange
from newton import metodo_newton

with open("output.txt", "w") as output:
    numero_sistemas = 5
    ns = [3, 5, 10]
    output.write("Gauss sem pivo\n\n")
    for i in range(len(ns)):
        for j in range(numero_sistemas):
            x = []
            y = []
            A, b = gerar_sistema(ns[i])

            x.append(eliminacao_gauss_sem_pivoteamento(A, b))
            y.append(np.linalg.solve(A, b))
            output.write(f"A solucao de m{ns[i]}, n{j + 1}:\n")
            
            for k in range(len(x)):
                for m in range(len(x[k])):
                    output.write(f"F(a{m}): {round(x[k][m], 4)},\tNumpy: {round(y[k][m], 4)},\tErro: {abs(round(x[k][m], 4) - round(y[k][m], 4))}\n")
                output.write("\n")

    output.write("Jacobi & Gauss Seidel\n\n")
    for i in range(len(ns)):
        for j in range(numero_sistemas):
            eqs, vars = gerar_equacoes(ns[i])
            resposta_j, iteracoes_j, numpy_jacobi = metodo_jacobi(eqs, vars)
            resposta_gs, iteracoes_gs, numpy_gauss = metodo_gauss_seidel(eqs, vars)
            for k in range(len(eqs)):
                output.write(f"\n{eqs[k]}\n\n")
                for l in range(len(vars)):
                    output.write(f"Jacobi, {vars[l]}: {round(resposta_j[l], 4)},\tNumpy:{round(numpy_jacobi[l], 4)},\tErro: {abs(round(resposta_j[l], 4) - round(numpy_jacobi[l], 4))}\n")
                    output.write(f"Gauss_Seidel, {vars[l]}: {round(resposta_gs[l], 4)},\tNumpy:{round(numpy_gauss[l], 4)},\tErro: {abs(round(resposta_gs[l], 4) - round(numpy_gauss[l], 4))}\n\n")
                output.write(f"Jacobi, iter: {iteracoes_j},\tGauss_Seidel, iter: {iteracoes_gs}\n\n\n")
            output.write(f"\n")

    output.write(f"\n\nLagrange & Newton\n\n")
    x = symbols('x')
    funcao = sin(2 * np.pi * x) + 0.2 * cos(4 * np.pi * x) + 0.1 * x

    n_pontos = [10, 20, 15]

    pontos_totais = []

    xs_totais = []
    ys_totais = []

    pontos_uniforme = np.linspace(0.0, 1.0, n_pontos[0])
    pontos_uniforme = [ round(float(p), 4) for p in pontos_uniforme]
    pontos_totais.extend(
        (p, round(funcao.subs(x, p), 4))
        for p in pontos_uniforme
    )
    xs_totais.extend(pontos_uniforme)
    ys_totais.extend(round(funcao.subs(x, p), 4) for p in pontos_uniforme)

    pontos_uniforme = np.linspace(0.0, 1.0, n_pontos[1])
    pontos_uniforme = [ round(float(p), 4) for p in pontos_uniforme]
    pontos_totais.extend(
        (p, round(funcao.subs(x, p), 4))
        for p in pontos_uniforme
    )
    xs_totais.extend(pontos_uniforme)
    ys_totais.extend(round(funcao.subs(x, p), 4) for p in pontos_uniforme)

    pontos_aleatorios = gerar_pontos_baseado_funcao(n_pontos[2], funcao)
    pontos_totais.extend(pontos_aleatorios)
    xs_totais.extend(p[0] for p in pontos_aleatorios)
    ys_totais.extend(round(funcao.subs(x, p[1]), 4) for p in pontos_aleatorios)

    p_x_lagrange = []
    p_x_newton = []
    xs_inter_lagrange = []
    xs_inter_newton = []

    print(xs_totais)
    print(ys_totais)
    for i in range(0, len(pontos_totais)):
        output.write(f"Pontos [{i}]: {pontos_totais[i]}\n")

    p_x_lagrange_temp, xs_inter_temp = metodo_lagrange(pontos_totais, xs_inter = xs_totais)
    p_x_lagrange.append(p_x_lagrange_temp)
    xs_inter_lagrange.extend(xs_inter_temp)

    p_x_newton_temp, xs_inter_temp = metodo_newton(xs_totais, ys_totais, xs_inter = xs_totais)
    p_x_newton.append(p_x_newton_temp)
    xs_inter_newton.extend(xs_inter_temp)
    output.write(f"\n\n")

    output.write(f"Equacao original: {funcao}\n")
    output.write(f"\nLagrange:\nEquacao P(x): {p_x_lagrange}\n\n")
    output.write(f"\nNewton:\nEquacao P(x): {p_x_newton}\n\n")

    for j in range(0, len(xs_inter_lagrange)):
        output.write(f"P_L({xs_totais[j]}): {round(xs_inter_lagrange[j], 4)},\tF_original: {pontos_totais[j][1]},\tErro: {abs(round(xs_inter_lagrange[j] - pontos_totais[j][1], 4))}\n")
    output.write("\n\n\n")
    for j in range(0, len(xs_inter_newton)):
        output.write(f"P_N({xs_totais[j]}): {round(xs_inter_newton[j], 4)},\tF_original: {pontos_totais[j][1]},\tErro: {abs(round(xs_inter_newton[j] - pontos_totais[j][1], 4))}\n")
    output.write("\n")


    output.write("\n\nInterpolacao e analise\n\n")
    xs_para_interpolar = np.linspace(0.0, 1.0, 1000)
    xs_para_interpolar = [ round(float(p), 4) for p in xs_para_interpolar]

    ys_depois_interpolar_Real = []
    ys_depois_interpolar_L = []
    ys_depois_interpolar_N = []

    ys_depois_interpolar_Real.extend((funcao.subs(x, xs_para_interpolar[i]) for i in range(0, len(xs_para_interpolar))))
    ys_depois_interpolar_L.extend((p_x_lagrange[0].subs(x, xs_para_interpolar[i]) for i in range(0, len(xs_para_interpolar))))
    ys_depois_interpolar_N.extend((p_x_newton[0].subs(x, xs_para_interpolar[i]) for i in range(0, len(xs_para_interpolar))))

    erros_abs_L = []
    erros_abs_N = []

    for i in range(0, len(xs_para_interpolar)):
        output.write(f"P_L({xs_para_interpolar[i]}): {round(ys_depois_interpolar_L[i], 4)},\tP_N({xs_para_interpolar[i]}): {round(ys_depois_interpolar_N[i], 4)},\tP_Original({xs_para_interpolar[i]}): {round(ys_depois_interpolar_Real[i], 4)}\n")
        output.write(f"Erro_L: {abs(round(ys_depois_interpolar_L[i] - ys_depois_interpolar_Real[i], 4))},\tErro_N: {abs(ys_depois_interpolar_N[i] - ys_depois_interpolar_Real[i])}\n\n")
        erros_abs_L.append(abs(round(ys_depois_interpolar_L[i] - ys_depois_interpolar_Real[i], 4)))
        erros_abs_N.append(abs(round(ys_depois_interpolar_N[i] - ys_depois_interpolar_Real[i], 4)))

    output.write(f"\n\n\nErro maximo Lagrange: {max(erros_abs_L)}\n")
    output.write(f"Erro maximo Newton: {max(erros_abs_N)}\n\n")

    output.write(f"Erro medio Lagrange: {round(sum(erros_abs_L)/len(xs_para_interpolar), 4)}\n")
    output.write(f"Erro medio Newton: {round(sum(erros_abs_N)/len(xs_para_interpolar), 4)}\n")
