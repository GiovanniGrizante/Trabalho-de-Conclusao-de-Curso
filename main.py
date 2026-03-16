import iema, ons, armazenar, ampl, sinteticas
import tratamento_rede, rede_neural
import graficos, graficos_aux

import tabulate as tab
import os, time

def gerar_tabela():
    conteudo = [
        ['Etapa 1', 'iema.py\nons.py\narmazenar.py', 'Processa os dados ONS e IEMA'],
        ['Etapa 2', 'ampl.py\nsinteticas.py', 'Encontra os coeficientes e emissões sintéticas.'],
        ['Etapa 3', 'tratamento_rede.py\nrede_neural.py', 'Trata os dados e executa o modelo de rede neural.'],
        ['Etapa 4', 'graficos.py\ngraficos_aux.py', 'Gera os gráficos das usinas.']
    ]
    
    cabecalho = ['Etapas', 'Arquivos', 'Descrição']
    
    print('=== Etapas do Pipeline ===')
    print(tab.tabulate(conteudo, headers = cabecalho, tablefmt = 'grid'))
    # Outros formatos: "simple", "plain", "grid", "fancy_grid", "pipe", "orgtbl", "jira", "presto", "pretty"
    
def perguntar_etapa(num):
    while True:
        resposta = input(f'Deseja executar a etapa {num}? (1 - Sim | 0 - Não): ')
        if resposta in ['0','1']:
            return resposta
        print('Entrada inválida. Por favor, insira (1 - Sim | 0 - Não)')
        time.sleep(4)
    
def executar_arq():
    
    dados = {
        'iema': None,
        'ons': None
    }
    
    etapas = {
        1: [lambda: dados.update({'iema': iema.main()}), 
            lambda: dados.update({'ons': ons.main(dados['iema'])}), 
            lambda: armazenar.main(dados['iema'], dados['ons'])],
        2: [lambda: ampl.main()],
        3: [lambda: sinteticas.main(), lambda: tratamento_rede.main()],
        4: [lambda: rede_neural.main()],
        5: [lambda: graficos.main(), lambda: graficos_aux.main()]
    }
    
    for num in etapas.keys():
        os.system('cls')
        gerar_tabela()
        resposta = perguntar_etapa(num)
        
        if resposta == '1':
            for func in etapas[num]:
                func()
                
executar_arq()