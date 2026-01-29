from amplpy import AMPL
import pandas as pd
import os
from iema import anos

def ampl_model(ons_ampl,iema_ampl,usina):

    # Inicializar o ambiente AMPL
    ampl = AMPL()

    # Definir o modelo AMPL usando ampl.eval
    ampl.eval(r'''
            
    set x;  # Quantidade de anos a serem estudados
    set y;  # Quantidade de horas do ano
              
    param HorasAno{x};  # Número de horas para cada ano, considerando bissextos
            
    param PG{x,y};
    param E{x};

    var alpha;
    var beta;
    var gamma;
    var omega;
    var mu;

    minimize Obj:
        (1 / card{x}) * sum {i in x} (sum {j in y: j <= HorasAno[i] and PG[i,j] != 0} ((alpha * PG[i,j]^2 + beta * PG[i,j] + gamma + omega * exp(mu * PG[i,j]) - E[i])^2) + 1e-12);
              
    subject to ALPHA: alpha >= 0;
    subject to BETA: beta >= 0;
    subject to GAMMA: gamma >= -omega;
    subject to OMEGA: omega >= 0;
    subject to MU: mu >= 0;
    ''')

    ampl.set['x'] = list(range(1,len(anos)+1))
    ampl.set['y'] = list(range(1, 8784 + 1))

    # Definindo o parâmetro 'HorasAno'
    horas_ano = {}
    for i, ano in enumerate(anos, start=1):
        horas_ano[i] = 8784 if int(ano) % 4 == 0 else 8760
    ampl.param['HorasAno'] = horas_ano

    for i, ano in enumerate(anos, start=1):
        for j, valor in enumerate(ons_ampl[i], start=1):
            # Certifique-se de que 'j' está dentro do intervalo de horas para o ano 'ano'
            if j <= horas_ano[i]:
                ampl.eval(f'let PG[{i},{j}] := {valor};')

    ampl.param['E'] = iema_ampl

    # Resolver o modelo
    ampl.setOption('solver','C:\\Users\\Giovanni\\Documents\\ipopt\\bin\\ipopt')
    ampl.solve(ampl_model)

    # Obter os valores das variáveis de decisão
    alpha = ampl.var['alpha'].value()
    beta = ampl.var['beta'].value()
    gamma = ampl.var['gamma'].value()
    omega = ampl.var['omega'].value()
    mu = ampl.var['mu'].value()

    print(alpha)
    print(beta)
    print(gamma)
    print(omega)
    print(mu)

    #Salvando os resultados
    # tab = pd.DataFrame({'Coeficientes':['Alpha','Beta','Gamma','Omega','Mu'],'Valores':[alpha,beta,gamma,omega,mu]})
    # tab.to_csv(f'{dir2}\\{usina}_{metodo_objetivo}_{pu_mw}_3_anos.csv',index=False)

    #Limpando o modelo para disponibilizar armazenamento na memória
    #del ampl_model