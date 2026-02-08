from amplpy import AMPL
import pandas as pd
import os
from natsort import natsorted

# Tempo médio de execução para 55 usinas: 2h40min

def ampl_model(usina):
    # Inicializar o ambiente AMPL
    ampl = AMPL()

    # Resetar tudo
    ampl.eval('reset;')
    
    # Configurar solver e opções para não exibir mensagens de log
    ampl.eval('option solver ipopt;')
    ampl.eval('option ipopt_options "print_level=0 sb=yes print_user_options=no";')

    # Carregar o arquivo .mod (modelo)
    ampl.read(os.path.join('AMPL', 'emissoes_CO2e.mod'))

    # Carregar o arquivo .dat (dados)
    ampl.read(os.path.join('AMPL', 'Usinas', usina, f'{usina}.dat'))

    # Carregar o arquivo .run (execução)
    ampl.eval(f'include "{os.path.join("AMPL", "emissoes_CO2e.run")}";')

    # Obter os valores das variáveis de decisão
    alpha = ampl.var['Alpha_est'].value()
    beta = ampl.var['Beta_est'].value()
    gamma = ampl.var['Gamma_est'].value()
    omega = ampl.var['Omega_est'].value()
    mu = ampl.var['Mu_est'].value()

    alpha_ant = ampl.var['Alpha_din_ant'].value()
    beta_ant = ampl.var['Beta_din_ant'].value()
    gamma_ant = ampl.var['Gamma_din_ant'].value()
    omega_ant = ampl.var['Omega_din_ant'].value()
    mu_ant = ampl.var['Mu_din_ant'].value()

    alpha_pos = ampl.var['Alpha_din_pos'].value()
    beta_pos = ampl.var['Beta_din_pos'].value()
    gamma_pos = ampl.var['Gamma_din_pos'].value()
    omega_pos = ampl.var['Omega_din_pos'].value()
    mu_pos = ampl.var['Mu_din_pos'].value()

    MSE_est = ampl.getObjective('MSE_est').value()
    MSE_din = ampl.getObjective('MSE_din').value()

    # Salvando os resultados em um DataFrame e exportando para CSV
    tab = pd.DataFrame({'Coeficientes':['Alpha','Beta','Gamma','Omega','Mu', 'MSE'],
                        'Valores Estáticos':[alpha,beta,gamma,omega,mu, MSE_est],
                        'Valores Dinâmicos Anteriores':[alpha_ant,beta_ant,gamma_ant,omega_ant,mu_ant, MSE_din],
                        'Valores Dinâmicos Posteriores':[alpha_pos,beta_pos,gamma_pos,omega_pos,mu_pos, MSE_din]})
    
    os.makedirs(f'Dados Tratados\\{usina}\\Coeficientes', exist_ok=True)
    tab.to_parquet(f'Dados Tratados\\{usina}\\Coeficientes\\{usina}.parquet',index=False)


# Definição dos dados a serem usado pelo modelo AMPL
for usina in natsorted(os.listdir(os.path.join('AMPL', 'Usinas'))):
    ampl_model(usina)