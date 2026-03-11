import os, pandas as pd
# import tensorflow as tf
# import h5py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from keras import backend as K  #Função para resetar a rede criada


# Ler a tabela mais recente do IEMA e filtrar os dados para as usinas presentes na pasta "Dados Tratados"
dir = 'IEMA/Tabelas'
iema_recente = pd.read_excel(os.path.join(dir, sorted(os.listdir(dir))[-1]))

colunas_remover = ['Município', 
                   'Geração [GWh]', 
                   'Emissões de Gases [tCO2]', 
                   'Taxa de Emissão [tCO2/GWh]']

if os.listdir('Dados Tratados')[0][:3] == 'UTE':
    iema_recente = iema_recente[iema_recente[f'CEG'].isin(os.listdir('Dados Tratados'))].reset_index(drop=True)
    colunas_remover.append('Usina')
else:
    iema_recente = iema_recente[iema_recente[f'Usina'].isin(os.listdir('Dados Tratados'))].reset_index(drop=True)
    colunas_remover.append('CEG')

iema_recente = iema_recente.drop(columns=colunas_remover)

# Converter os erros das colunas numéricas para NaN
for coluna in ['Potência Instalada', 'Fator de Capacidade [%]', 'Eficiência Energética [%]']:
    iema_recente[coluna] = pd.to_numeric(iema_recente[coluna], errors='coerce')

print(iema_recente.head())

# # Aplicar OneHotEncoder (Transformação das colunas categóricas em numéricas) e manter as colunas numéricas
# onehot = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),['Combustível','Ciclo de Operação'])],remainder='passthrough')
# data = onehot.fit_transform(iema_recente)

# # Aplicar StandardScaler (Padronização das colunas numéricas) e manter as colunas resultantes do OneHotEncoder
# scaler = ColumnTransformer(transformers=[('scaler',StandardScaler(),['Potência Instalada', 'Fator de Capacidade [%]', 'Eficiência Energética [%]'])],remainder='passthrough')
# data = scaler.fit_transform(data)

# # Obter os nomes das colunas resultantes do OneHotEncoder e combinar com as colunas restantes
# nomes_onehot = onehot.named_transformers_['encoder'].get_feature_names_out(['Combustível', 'Ciclo de Operação'])
# colunas_restantes = iema_recente.drop(columns=['Combustível', 'Ciclo de Operação']).columns
# colunas = list(nomes_onehot) + list(colunas_restantes)

# iema_recente = pd.DataFrame(data,columns=colunas)








# Mesclagem dos dados de geração com os dados de emissão
# for usina in os.listdir('Dados Tratados'):
    


# # Tratamento específico dos dados

# usinas_estudadas = ['Baixada Fluminense','Norte Fluminense','Santa Cruz','Termorio']
# usina_serie = tab_trat['Usina']

# if not usinas_estudadas:
#     raise ValueError(f'Nenhuma usina a ser estudada.')
# elif not all(usina in usinas for usina in usinas_estudadas):
#     raise ValueError(f'Usinas não listadas.')

# temp = []
# for usina in usinas_estudadas:

#     dir = f'G:\\Meu Drive\\Documentos UFSCar\\Iniciação Científica (Giovanni Grizante)\\Python\\Termelétricas (PANDAS)\\{usina}\\Emissões Sintéticas (Todos os anos)'
#     arq = sorted([f for f in os.listdir(dir) if f.endswith('.csv')])

#     for arquivo in arq:
#         dir_comp = os.path.join(dir,arquivo)
#         temp.append(pd.read_csv(dir_comp))

#     emi = pd.concat(temp)


# def preservar_usinas(x):
#     return ', '.join(sorted(set(x)))

# def verificar_combustiveis(x):
#     combustiveis_unicos = sorted(set(x))
#     if len(combustiveis_unicos) > 1:
#         raise ValueError(f'Erro: Mais de um tipo de combustível encontrado: {', '.join(combustiveis_unicos)}')
#     return combustiveis_unicos[0]
                         
# emi = emi.groupby(['Dia-Hora']).agg({
#     'Usina': preservar_usinas,
#     'Emissão': 'sum',             # Soma as emissões
#     'Combustível': verificar_combustiveis
# })

# x = tab_trat[['ohe1','ohe2','ohe3','ohe4','ohe5','ohe6','ohe7','ohe8','Geração','Potência Instalada']]

# # Definição das variáveis de treino e teste
# sc = StandardScaler() # Transforma variáveis categóricas em numéricas
# x = sc.fit_transform(x)

# x = pd.DataFrame({'Coluna1': x[:, 0], 
#                   'Coluna2': x[:, 1], 
#                   'Coluna3': x[:, 2], 
#                   'Coluna4': x[:, 3], 
#                   'Coluna5': x[:, 4], 
#                   'Coluna6': x[:, 5], 
#                   'Coluna7': x[:, 6], 
#                   'Coluna8': x[:, 7], 
#                   'Geração': x[:, 8], 
#                   'Potência Instalada': x[:, 9], 
#                   'Usina': usina_serie,
#                   'Dia-Hora': tab_trat['Dia-Hora']})

# x = x.loc[x['Usina'].isin(usinas_estudadas)]

# x = x.groupby(['Dia-Hora']).agg({
#     'Usina': preservar_usinas,
#     'Coluna1':'sum',
#     'Coluna2':'sum',
#     'Coluna3':'sum',
#     'Coluna4':'sum',
#     'Coluna5':'sum',
#     'Coluna6':'sum',
#     'Coluna7':'sum',
#     'Coluna8':'sum',
#     'Geração':'sum',
#     'Potência Instalada': 'sum'
# })

# x = x.drop(['Usina'],axis=1)
# y = emi['Emissão']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)