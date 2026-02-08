import pandas as pd
import os
from natsort import natsorted
from iema import horas, anos, usinas_iema, metodo, relacao

# Funções auxiliares

def formatacao_dataframe(ons_total, metodo, usinas_iema):

    # Remover pontos e os últimos dois dígitos de cada valor na coluna CEG
    ons_total[relacao[metodo]] = ons_total[relacao[metodo]].astype(str).str.replace('.', '', regex=False).str[:15]

    # Filtrar o DataFrame para manter apenas as usinas que possuem dados na IEMA
    ons_total = ons_total[ons_total[relacao[metodo]].isin(usinas_iema)]

    # Remover colunas desnecessárias com condições corretas
    colunas_para_remover = ['id_subsistema',
                            'nom_subsistema',
                            'nom_estado',
                            'cod_modalidadeoperacao',
                            'id_estado',
                            'nom_tipousina',
                            'nom_tipocombustivel',
                            'id_ons']
    
    colunas_para_renomear = {'din_instante':'Ano',
                             'nom_usina':'Usina',
                             'ceg':'CEG',
                             'val_geracao':'Geração'}
    
    if metodo == 'CEG':
        colunas_para_remover.append('nom_usina')
        colunas_para_renomear.pop('nom_usina', None)
    if metodo == 'Usina':
        colunas_para_remover.append('ceg')
        colunas_para_renomear.pop('ceg', None)

    ons_total = ons_total.drop(columns=colunas_para_remover)

    # Trocar o nome das colunas restantes
    ons_total = ons_total.rename(columns=colunas_para_renomear)

    # Converter a coluna de geração para float, substituindo valores vazios por 0
    ons_total['Geração'] = pd.to_numeric(ons_total['Geração'], errors='coerce').fillna(0)

      

    return ons_total


# Este código acessa os dados da ONS armazenados em arquivos PARQUET, organiza as informações e separa os dados por usina térmica, 
# considerando apenas aquelas que possuem dados de emissão na IEMA para todos os anos estudados.

dir_ons = "G:\\Meu Drive\\Documentos UFSCar\\TCC\\ONS"

ons = {}    # Variável para armazenar os dados das usinas térmicas
ons_total = []  # Variável para armazenar todos os dados lidos

# inicio = time.time()

# Acessar o drive e armazenar os dados
for ano in anos:
    dir_ano = os.path.join(dir_ons, ano)

    # Listar e ordenar os arquivos PARQUET dentro do diretório do ano usando natsorted
    arq_parquet = natsorted([f for f in os.listdir(dir_ano) if f.endswith('.parquet')])

    for arquivo in arq_parquet:
        # Ler os arquivos PARQUET e armazená-los em uma lista de DataFrames
        df_ano = pd.read_parquet(os.path.join(dir_ano, arquivo))
        # Concatenar os DataFrames do ano e adicionar à lista total
        ons_total.append(df_ano)


# Concatenar todos os DataFrames dos anos em um único DataFrame final
ons_total = pd.concat(ons_total, ignore_index=True)

# Atualizar o DataFrame com o valor retornado pela função
ons_total = formatacao_dataframe(ons_total, metodo, usinas_iema)

# Calcular os anos para os deltas
anos_int = list(map(int, anos))   # Transformar os itens da lista "anos" para o tipo int
ano_anterior = str(min(anos_int) - 1)
ano_posterior = str(max(anos_int) + 1)

anos_delta = sorted(list({ano_anterior, ano_posterior}))

# Acessar os dados das usinas térmicas e armazená-los em um dicionário, considerando apenas as usinas que possuem dados de emissão na IEMA para todos os anos estudados
for usina in usinas_iema:
    if len(ons_total.loc[ons_total[metodo] == usina]) == sum(horas.values()):   # Verificar se a quantia de horas do DataFrame corresponde ao total de horas dos anos estudados

        ons[usina] = ons_total.loc[ons_total[metodo] == usina].reset_index(drop=True)    # Armazenar os dados de cada usina no dicionário
        ons[usina].drop(columns=['CEG'], inplace=True)  # Remover a coluna do método usado para separar as usinas térmicas

        geracao_delta = []  # Lista para armazenar os valores de geração dos anos vizinhos

        # Acessar os dados dos anos vizinhos para calcular os deltas
        for ano in anos_delta:
            dir_delta = os.path.join(dir_ons, ano)

            # Listar e ordenar os arquivos PARQUET dentro do diretório do ano usando natsorted
            arq_parquet = natsorted([f for f in os.listdir(dir_delta) if f.endswith('.parquet')])

            # Ler o primeiro arquivo PARQUET e armazená-lo
            try:
                df_delta = pd.read_parquet(os.path.join(dir_delta, arq_parquet[0]))
            except IndexError:
                print(f'Arquivo PARQUET não encontrado para o ano {ano}.')
                exit()

            try:
                if metodo == 'Usina':
                    df_delta = df_delta.loc[df_delta['nom_usina'] == usina]
                else:
                    df_delta['ceg'] = df_delta['ceg'].astype(str).str.replace('.', '', regex=False).str[:15]  # Remover pontos e os últimos dois dígitos de cada valor na coluna CEG
                    df_delta = df_delta.loc[(df_delta['ceg'] == usina)]

                data_geracao = df_delta['din_instante'].iloc[-1] if ano == ano_anterior else df_delta['din_instante'].iloc[0]
                data_geracao = data_geracao.strftime('%Y-%m-%d %H:%M:%S')  # Converte para string no formato esperado

                if data_geracao.startswith(f'{ano}-12-31') and data_geracao.endswith('23:00:00'):
                    valor_geracao = pd.to_numeric(df_delta['val_geracao'].iloc[-1], errors='coerce')
                elif data_geracao.startswith(f'{ano}-01-01') and data_geracao.endswith('00:00:00'):
                    valor_geracao = pd.to_numeric(df_delta['val_geracao'].iloc[0], errors='coerce')

                geracao_delta.append(valor_geracao)
            except IndexError:
                geracao_delta.append(0)
                print(f'Usina {usina} não encontrada para o ano {ano}. Adicionando valor 0.')

        # Gerar as colunas 'Delta menos' e 'Delta mais'
        ons[usina]['Delta menos'] = ons[usina]['Geração'].diff().round(3)
        ons[usina]['Delta mais'] = (ons[usina]['Geração'].diff(-1) * -1).round(3)
            
        # Calcular os valores corretos para os deltas nas bordas
        ons[usina].loc[0,'Delta menos'] = round(ons[usina].loc[0,'Geração'] - geracao_delta[0], 3)
        ons[usina].loc[ons[usina].index[-1],'Delta mais'] = round(geracao_delta[1] - ons[usina].loc[ons[usina].index[-1],'Geração'], 3)

        # Ajustar o formato da coluna 'Ano' para manter apenas o ano
        ons[usina]['Ano'] = ons[usina]['Ano'].astype(str).str[:4]

        # Criar um índice relacionado com os valores da lista 'horas'
        indices = []
        for h in horas.values():
            indices.extend(list(range(h)))
        ons[usina]['Índice'] = indices[:len(ons[usina])]