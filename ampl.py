import sys
from amplpy import AMPL
import pandas as pd
import os
from natsort import natsorted

def ampl_model(dir, usina):
    # Inicializar o ambiente AMPL
    ampl = AMPL()

    # Carregar o arquivo .mod (modelo)
    ampl.read(os.path.join(dir, 'emissoes_CO2e.mod'))

    # Carregar o arquivo .dat (dados)
    ampl.read(os.path.join(dir, 'Usinas', usina, f'{usina}.dat'))

    # Definir o solver a ser utilizado
    ampl.setOption('solver','D:\\Ipopt\\bin\\ipopt')
    #ampl.eval("option solver ipopt;")

    # Carregar o arquivo .run (execução)
    ampl.eval(f'include "{os.path.join(dir,"emissoes_CO2e.run")}";')

    # Obter os valores das variáveis de decisão
    # alpha = ampl.var['alpha'].value()
    # beta = ampl.var['beta'].value()
    # gamma = ampl.var['gamma'].value()
    # omega = ampl.var['omega'].value()
    # mu = ampl.var['mu'].value()

    # print(alpha)
    # print(beta)
    # print(gamma)
    # print(omega)
    # print(mu)

    #Salvando os resultados
    # tab = pd.DataFrame({'Coeficientes':['Alpha','Beta','Gamma','Omega','Mu'],'Valores':[alpha,beta,gamma,omega,mu]})
    # tab.to_csv(f'{dir2}\\{usina}_{metodo_objetivo}_{pu_mw}_3_anos.csv',index=False)

    #Limpando o modelo para disponibilizar armazenamento na memória
    #del ampl_model




# Definição dos diretórios (Base e AMPL)
dir = f'G:\\Meu Drive\\Documentos UFSCar\\TCC\\AMPL'

# Definição dos dados a serem usado pelo modelo AMPL
for usina in natsorted(os.listdir(os.path.join(dir, 'Usinas'))):
    ampl_model(dir, usina)
    sys.exit()