import tabulate as tab
import os, time

from Arquivos import iema, ons, armazenar, ampl, sinteticas
from Arquivos import tratamento_rede, rede_neural
from Arquivos import graficos


def gerar_tabela():
    conteudo = [
        ['Etapa 1', 'iema.py\nons.py\narmazenar.py', 'Processa os dados ONS e IEMA.'],
        ['Etapa 2', 'ampl.py', 'Encontra os coeficientes\n(Executar o arquivo p/ multiprocessamento).'],
        ['Etapa 3', 'sinteticas.py\ntratamento_rede.py', 'Gera as emissões sintéticas e trata os dados.'],
        ['Etapa 4', 'rede_neural.py', 'Treina o modelo de rede neural (GRU + Dense).'],
        ['Etapa 5', 'graficos.py\ngraficos_aux.py', 'Gera os resultados gerais e por usina da RN.'],
        ['Etapa 6', 'previsao.py', 'Executa o modelo para previsão de emissões']
    ]
    
    cabecalho = ['Etapas', 'Arquivos', 'Descrição']
    
    print('=== Etapas do Pipeline ===')
    print(tab.tabulate(conteudo, headers = cabecalho, tablefmt = 'grid'))
    # Outros formatos: "simple", "plain", "grid", "fancy_grid", "pipe", "orgtbl", "jira", "presto", "pretty"
   
def executar_etapas():
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
        5: [lambda: graficos.main()]
    }
    
    return etapas
    
def perguntar_etapas(etapas):
    while True:
        os.system('cls')
        gerar_tabela()
        num = input(f'Digite a etapa a ser executada (Em branco para todas): ')
        num = int(num) if num.isnumeric() else num
        
        if num in etapas.keys():
            for func in etapas[num]:
                func()
        elif num == '':
            for num in etapas.keys():
                for func in etapas[num]:
                    func()
        else:
            print('Etapa inválida.')
            time.sleep(4)
    
if __name__ == '__main__':
    etapas = executar_etapas()
    perguntar_etapas(etapas)