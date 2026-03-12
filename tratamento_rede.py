import os, pandas as pd, multiprocessing, sys, numpy as np
# import tensorflow as tf
# import h5py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from keras import backend as K  #Função para resetar a rede criada


def tratamento_dados(iema_recente):
    # Mesclagem e normalização dos dados por usina
    for usina in os.listdir('Dados Tratados'):
        geracao = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Dados Externos', f'ONS.parquet'))
        emissao = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Emissões Sintéticas', f'Horárias.parquet'))
        
        try:
            unitarios = iema_recente[iema_recente['Usina'] == usina].drop(columns=['Usina']).reset_index(drop=True)
        except KeyError:
            unitarios = iema_recente[iema_recente['CEG'] == usina].drop(columns=['CEG']).reset_index(drop=True)
            
        # Unificação dos dados de geração, emissão e dados unitários
        tabela = pd.merge(geracao, emissao, on=None, how='left').merge(unitarios, on=None, how='cross')
        
        # Remover colunas de delta. Caso desejar incluir, comentar linha abaixo.
        tabela = tabela.drop(columns=['Delta menos', 'Delta mais'])
        
        
        # == FEATURES DE TRANSIÇÃO ==
        # Features importantes para a rede neural tratar as transições como um processo, e não como um estado fixo.
        
        # Duração da transição
        # Flag de transição (1 se estiver em transição, 0 caso contrário)
        transicao_list = list((tabela['Categoria de geração'] == 0).astype(int))
        duracao = np.zeros_like(transicao_list, dtype=int)
        contador = 0
        for i in range(len(transicao_list)):
            if transicao_list[i]:
                contador += 1
            else:
                contador = 0
            duracao[i] = contador
        tabela['Duração da Transição'] = duracao
        
        # Fase da transição
        fase = np.zeros_like(transicao_list, dtype=int)
        for i in range(len(transicao_list)):
            if transicao_list[i]:  # em transição
                inicio = (i == 0) or (not transicao_list[i-1])  # Verificar se é o início da transição ou se é a primeira linha
                fim = (i == len(transicao_list) - 1) or (not transicao_list[i+1])  # Verificar se é o fim da transição ou se é a última linha
                
                if inicio and fim:
                    fase[i] = 1  # Transição de apenas uma hora (início e fim ao mesmo tempo)
                elif inicio:  # hora anterior não era transição
                    fase[i] = 1  # INÍCIO
                elif fim:  # próxima hora não será transição
                    fase[i] = 3  # FIM
                else:
                    # MEIO: pode ser hora 2 de 3, ou hora 2 de 4, etc.
                    fase[i] = 2  # MEIO (qualquer hora que não é início nem fim)
        tabela['Fase da Transição'] = fase
        
        # Definir as colunas categóricas e numéricas para o ColumnTransformer
        categoricas = ['Categoria de geração']
        numericas = ['Geração', 'Duração da Transição', 'Fase da Transição']  # Adicionar outras colunas numéricas, se necessário
        
        # Transformação da coluna categórica "Categoria de geração" em numérica e normalização da coluna numérica "Geração"
        transf = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(sparse_output=False), categoricas),
            ('scaler', StandardScaler(), numericas)],
                                   remainder='passthrough',  # Mantém constant_cols inalteradas
                                   verbose_feature_names_out=False  # Para nomes mais limpos
                                   )
        
        data = transf.fit_transform(tabela)
        
        # Geração da tabela final
        tabela = pd.DataFrame(data, columns=transf.get_feature_names_out())
        
        # Salvar a tabela unificada em formato parquet
        os.makedirs(os.path.join('Dados Tratados', usina, 'Rede Neural'), exist_ok=True)
        tabela.to_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados_Entrada.parquet'))
        sys.exit()
        

# Ler a tabela mais recente do IEMA e filtrar os dados para as usinas presentes na pasta "Dados Tratados"
iema_recente = pd.read_excel(os.path.join('IEMA/Tabelas', sorted(os.listdir('IEMA/Tabelas'))[-1]))

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

# Aplicar OneHotEncoder (Transformação das colunas categóricas em numéricas) e manter as colunas numéricas
# É necessário aplicar esse processo para garantir que sejam analisadas todas as combinações possíveis.
# Após aplicar os dados em cada usina específica, as combinações se perdem.
onehot = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),['Combustível','Ciclo de Operação'])],remainder='passthrough')
data = onehot.fit_transform(iema_recente)

# Obter os nomes das colunas resultantes do OneHotEncoder e combinar com as colunas restantes
nomes_onehot = onehot.named_transformers_['encoder'].get_feature_names_out(['Combustível', 'Ciclo de Operação'])
colunas_restantes = iema_recente.drop(columns=['Combustível', 'Ciclo de Operação']).columns
colunas = list(nomes_onehot) + list(colunas_restantes)

iema_recente = pd.DataFrame(data,columns=colunas)
tratamento_dados(iema_recente)



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