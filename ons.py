import pandas as pd
import os
from natsort import natsorted
from iema import anos, horas, usinas_iema, metodo

# Este código acessa os dados da ONS armazenados em arquivos PARQUET, organiza as informações e separa os dados por usina térmica, 
# considerando apenas aquelas que possuem dados de emissão na IEMA para todos os anos estudados.

dir = "G:\\Meu Drive\\Documentos UFSCar\\TCC\\ONS"

ons = {}    # Variável para armazenar os dados das usinas térmicas
ons_total = []  # Variável para armazenar todos os dados lidos 


# Acessar o drive e armazenar os dados
for ano in anos:
    dir_ano = os.path.join(dir, ano)

    # Listar e ordenar os arquivos PARQUET dentro do diretório do ano usando natsorted
    arq_parquet = natsorted([f for f in os.listdir(dir_ano) if f.endswith('.parquet')])

    for arquivo in arq_parquet:
        # Ler os arquivos PARQUET e armazená-los em uma lista de DataFrames
        df_ano = pd.read_parquet(os.path.join(dir_ano, arquivo))
        # Concatenar os DataFrames do ano e adicionar à lista total
        ons_total.append(df_ano)


# Concatenar todos os DataFrames dos anos em um único DataFrame final
ons_total = pd.concat(ons_total, ignore_index=True)

# Separação em usinas térmicas
ons_total = ons_total.loc[ons_total['nom_tipousina']=='TÉRMICA']


# Remover colunas desnecessárias
ons_total = ons_total.drop(columns=['id_subsistema',
                        'nom_subsistema',
                        'nom_estado',
                        'cod_modalidadeoperacao',
                        'id_estado',
                        'id_ons'])

# Trocar o nome das colunas restantes
ons_total = ons_total.rename(columns={'din_instante':'Dia-Hora',
                          'nom_tipocombustivel':'Combustível',
                          'nom_usina':'Usina',
                          'ceg':'CEG',
                          'nom_tipousina':'Tipo',
                          'val_geracao':'Geração'})

# Converter a coluna 'Geração' para float, substituindo valores vazios por 0
ons_total['Geração'] = pd.to_numeric(ons_total['Geração'], errors='coerce').fillna(0)

ons_total['CEG'] = ons_total['CEG'].astype(str).str.replace('.', '', regex=False).str[:-2]  # Remover pontos e os últimos dois dígitos de cada valor na coluna CEG

ons_total = ons_total[ons_total[metodo].isin(usinas_iema)]   # Considerar apenas usinas térmicas que possuem dados de emissão na IEMA

# Calcular os valores para os deltas
anos_int = list(map(int, anos))   # Transformar os itens da lista "anos" para o tipo int
ano_anterior = str(min(anos_int) - 1)
ano_posterior = str(max(anos_int) + 1)

anos_delta = set(list({ano_anterior, ano_posterior}))

for usina in usinas_iema:
    if len(ons_total.loc[ons_total[metodo] == usina]) == sum(horas):   # Verificar se a quantia de horas do DataFrame corresponde ao total de horas dos anos estudados

        ons[usina] = ons_total.loc[ons_total[metodo] == usina].reset_index(drop=True)    # Armazenar os dados de cada usina no dicionário
        geracao_delta = []  # Lista para armazenar os valores de geração dos anos vizinhos

        # Acessar os dados dos anos vizinhos para calcular os deltas
        for ano in anos_delta:
            dir_delta = os.path.join(dir, ano)

            # Listar e ordenar os arquivos PARQUET dentro do diretório do ano usando natsorted
            arq_parquet = natsorted([f for f in os.listdir(dir_delta) if f.endswith('.parquet')])

            # Ler o primeiro arquivo PARQUET e armazená-lo
            try:
                df_delta = pd.read_parquet(os.path.join(dir_delta, arq_parquet[0]))
            except IndexError:
                print(f'Arquivo PARQUET não encontrado para o ano {ano}.')
                exit()

            # Obter o valor de geração da usina nos anos vizinhos
            try:
                valor_geracao = pd.to_numeric(df_delta.loc[(df_delta['nom_usina'] == usina) | (df_delta['ceg'] == usina), 'val_geracao'].iloc[-1], errors='coerce') # Converter para numérico
                geracao_delta.append(valor_geracao)
            except IndexError:
                geracao_delta.append(0)
                print(f'Usina {usina} não encontrada para o ano {ano}. Adicionando valor 0.')

        # Gerar as colunas 'Delta menos' e 'Delta mais'
        ons[usina]['Delta menos'] = ons[usina]['Geração'].diff()
        ons[usina]['Delta mais'] = ons[usina]['Geração'].diff(-1) * -1
            
        # Calcular os valores corretos para os deltas nas bordas
        ons[usina].loc[0,'Delta menos'] = ons[usina].loc[0,'Geração'] - geracao_delta[0]
        ons[usina].loc[ons[usina].index[-1],'Delta mais'] = geracao_delta[1] - ons[usina].loc[ons[usina].index[-1],'Geração']