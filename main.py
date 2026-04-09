import tabulate as tab
import os, time, sys

from Arquivos import iema, ons, armazenar, ampl, sinteticas
from Arquivos import tratamento_rede, rede_neural
from Arquivos import graficos


def gerar_tabela(previsao):
    if previsao == '1':
        conteudo = [
            ['Etapa 1', 'iema.py\nons.py\narmazenar.py', 'Processa os dados ONS e IEMA.'],
            ['Etapa 2', 'ampl.py', 'Encontra os coeficientes\n(Executar o arquivo p/ multiprocessamento).'],
            ['Etapa 3', 'sinteticas.py\ntratamento_rede.py', 'Gera as emissões sintéticas e trata os dados.'],
            ['Etapa 4', 'rede_neural.py', 'Treina o modelo de rede neural (GRU + Dense).'],
            ['Etapa 5', 'graficos.py', 'Gera os resultados gerais e por usina da RN.'],
            ['Etapa 6', 'previsao.py', 'Executa o modelo para previsão de emissões']
        ]
    else:
        conteudo = [
            ['Etapa 1', 'iema.py\nons.py\narmazenar.py', 'Processa os dados ONS e IEMA.'],
            ['Etapa 2', 'tratamento_rede.py', 'Trata os dados para a RN.'],
            ['Etapa 3', 'rede_neural.py', 'Treina o modelo de rede neural (GRU + Dense).'],
            ['Etapa 4', 'graficos.py', 'Gera os resultados gerais e por usina da RN.'],
            ['Etapa 5', 'previsao.py', 'Executa o modelo para previsão de emissões']
        ]
        
    cabecalho = ['Etapas', 'Arquivos', 'Descrição']
    
    print('=== Etapas do Pipeline ===')
    print(tab.tabulate(conteudo, headers = cabecalho, tablefmt = 'grid'))
    # Outros formatos: "simple", "plain", "grid", "fancy_grid", "pipe", "orgtbl", "jira", "presto", "pretty"
   
def executar_etapas(previsao):
    # Execução da etapa 1 (Arquivos que dependem de outros arquivos)
    def etapa1():
        iema = iema.main()
        ons_resultado = ons.main(iema)
        armazenar.main(iema, ons_resultado)
        
    etapas = {
        1: [etapa1]
    }
    
    if previsao == '1':
        etapas.update({
            2: [lambda: ampl.main()],
            3: [lambda: sinteticas.main(), lambda: tratamento_rede.main(previsao)],
            4: [lambda: rede_neural.main(previsao)],
            5: [lambda: graficos.main(previsao)]
        })
    elif previsao == '2':
        etapas.update({
            2: [lambda: tratamento_rede.main(previsao)],
            3: [lambda: rede_neural.main(previsao)],
            4: [lambda: graficos.main(previsao)]
        })
    
    return etapas
    
def perguntar_etapas(previsao, etapas):
    while True:
        os.system('cls')
        gerar_tabela(previsao)
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
    while True:
        os.system('cls')
        previsao = input('Qual método de previsão? (1 - Minimização | 2 - RN Híbrida): ')
        if previsao in ['1', '2']:
            break
    etapas = executar_etapas(previsao)
    perguntar_etapas(previsao, etapas)