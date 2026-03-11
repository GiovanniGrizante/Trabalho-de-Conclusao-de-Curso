from amplpy import AMPL
import pandas as pd
import os, time
import multiprocessing

# Tempo médio de execução para 55 usinas: 2h40min

def ampl_model(usina):
    # Inicializar o ambiente AMPL
    ampl = AMPL()

    # Resetar tudo
    ampl.eval('reset;')
    
    # Configurar solver e opções para não exibir mensagens de log
    ampl.eval('option solver ipopt;')
    ampl.eval('option ipopt_options "print_level=0 sb=yes print_user_options=no max_iter=10000";')

    # Carregar o arquivo .mod (modelo)
    ampl.read(os.path.join('AMPL', 'emissoes_CO2e.mod'))

    # Carregar o arquivo .dat (dados)
    ampl.read(os.path.join('AMPL', 'Usinas', usina, f'{usina}.dat'))

    # Carregar o arquivo .run (execução)
    ampl.eval(f'include "{os.path.join("AMPL", "emissoes_CO2e.run")}";')

    # Obter os valores das variáveis de decisão
    alpha = round(ampl.var['Alpha_est'].value(), 3)
    beta = round(ampl.var['Beta_est'].value(), 3)
    gamma = round(ampl.var['Gamma_est'].value(), 3)
    omega = round(ampl.var['Omega_est'].value(), 3)
    mu = round(ampl.var['Mu_est'].value(), 3)

    alpha_ant = round(ampl.var['Alpha_din_ant'].value(), 3)
    beta_ant = round(ampl.var['Beta_din_ant'].value(), 3)
    gamma_ant = round(ampl.var['Gamma_din_ant'].value(), 3)
    omega_ant = round(ampl.var['Omega_din_ant'].value(), 3)
    mu_ant = round(ampl.var['Mu_din_ant'].value(), 3)

    alpha_pos = round(ampl.var['Alpha_din_pos'].value(), 3)
    beta_pos = round(ampl.var['Beta_din_pos'].value(), 3)
    gamma_pos = round(ampl.var['Gamma_din_pos'].value(), 3)
    omega_pos = round(ampl.var['Omega_din_pos'].value(), 3)
    mu_pos = round(ampl.var['Mu_din_pos'].value(), 3)

    MSE_est = round(ampl.getObjective('MSE_est').value(), 3)
    MSE_din = round(ampl.getObjective('MSE_din').value(), 3)

    # Salvando os resultados em um DataFrame e exportando para CSV
    tab = pd.DataFrame({'Coeficientes':['Alpha','Beta','Gamma','Omega','Mu', 'MSE'],
                        'Valores Estáticos':[alpha,beta,gamma,omega,mu, MSE_est],
                        'Valores Dinâmicos Anteriores':[alpha_ant,beta_ant,gamma_ant,omega_ant,mu_ant, MSE_din],
                        'Valores Dinâmicos Posteriores':[alpha_pos,beta_pos,gamma_pos,omega_pos,mu_pos, MSE_din]})
    
    os.makedirs(f'Dados Tratados\\{usina}\\Emissões Sintéticas', exist_ok=True)
    tab.to_parquet(f'Dados Tratados\\{usina}\\Emissões Sintéticas\\Coeficientes.parquet',index=False)

    
# Para que o multiprocessamento seja ativado, é necessário que o código seja executado dentro do arquivo principal (main)
if __name__ == '__main__':
    
    # Determinar número de threads a serem utilizadas
    num_nucleos = int(input(f'Número de threads a serem utilizadas (máximo {multiprocessing.cpu_count() - 1}): '))

    if num_nucleos < 1 or num_nucleos > multiprocessing.cpu_count() - 1:
        raise ValueError(f'Número de threads deve ser entre 1 e {multiprocessing.cpu_count() - 1}.')


    usinas = os.listdir('Dados Tratados')

    # Lista para armazenar os processos
    processos = []

    # Índice para controlar qual usina será processada
    indice_usina = 0

    # Enquanto houver usinas para processar
    while indice_usina < len(usinas):
        # Iniciar novos processos até atingir o limite de núcleos
        while len(processos) < num_nucleos and indice_usina < len(usinas):
            usina_atual = usinas[indice_usina]
            
            # Criar e iniciar processo para a usina atual
            p = multiprocessing.Process(target=ampl_model, args=(usina_atual,))
            processos.append(p)
            p.start()
            
            indice_usina += 1
        
        # Verificar processos que já terminaram
        for p in processos[:]:  # Iterar sobre cópia da lista
            if not p.is_alive():
                p.join()  # Garantir que o processo foi finalizado
                processos.remove(p)  # Remover da lista de processos ativos
        
        # Pequena pausa para evitar uso excessivo de CPU na verificação
        if len(processos) >= num_nucleos:
            time.sleep(0.1)

    # Aguardar todos os processos restantes terminarem
    for p in processos:
        p.join()