from amplpy import AMPL
import pandas as pd
import os, time
import multiprocessing

# Tempo médio de execução para 55 usinas em 1 thread: 2h40min

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

    solve_message = ampl.get_value('solve_message')
    if 'Invalid number in NLP function' in str(solve_message):
        return None
    
    # Obter os valores das variáveis de decisão
    alpha = round(ampl.var['Alpha_est'].value(), 3)
    beta = round(ampl.var['Beta_est'].value(), 3)
    gamma = round(ampl.var['Gamma_est'].value(), 3)
    omega = round(ampl.var['Omega_est'].value(), 3)
    mu = round(ampl.var['Mu_est'].value(), 3)

    MSE_est = round(ampl.getObjective('MSE_est').value(), 3)

    # Salvando os resultados em um DataFrame e exportando para CSV
    tab = pd.DataFrame({'Coeficientes':['Alpha','Beta','Gamma','Omega','Mu', 'MSE'],
                        'Valores Estáticos':[alpha,beta,gamma,omega,mu, MSE_est]})
    
    os.makedirs(f'Usinas\\{usina}\\Minimização\\Emissões', exist_ok=True)
    tab.to_parquet(f'Usinas\\{usina}\\Minimização\\Emissões\\Coeficientes.parquet',index=False)

def main(multiprocessamento=False):
    # Para que o multiprocessamento seja ativado, é necessário que o código seja executado dentro do arquivo principal (main)
    if multiprocessamento:
        
        # Determinar número de threads a serem utilizadas
        num_nucleos = int(input(f'Número de threads a serem utilizadas (máximo {multiprocessing.cpu_count() - 1}): '))

        if num_nucleos < 1 or num_nucleos > multiprocessing.cpu_count() - 1:
            raise ValueError(f'Número de threads deve ser entre 1 e {multiprocessing.cpu_count() - 1}.')

        usinas = os.listdir('Usinas')

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

    else:
        print('Para ativar o multiprocessamento, execute o arquivo ampl.py')
        for usina in os.listdir('Usinas'):
            ampl_model(usina)
            
if __name__ == '__main__':
    main(multiprocessamento=True)