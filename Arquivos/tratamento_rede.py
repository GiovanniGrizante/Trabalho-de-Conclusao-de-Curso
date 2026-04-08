import os, pandas as pd, numpy as np, sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tratar os dados, aplicando OneHotEncoder e normalizando. Salva os arquivos tratados.
def tratamento_dados(iema_recente, previsao):
    # Mesclagem e normalização dos dados por usina
    for usina in os.listdir('Usinas'):
        geracao = pd.read_parquet(os.path.join('Usinas', usina, 'Dados Externos', f'ONS.parquet'))
        emissao = pd.read_parquet(os.path.join('Usinas', usina, 'Minimização', 'Emissões', f'Horárias.parquet'))
        
        try:
            unitarios = iema_recente[iema_recente['Usina'] == usina].drop(columns=['Usina']).reset_index(drop=True)
        except KeyError:
            unitarios = iema_recente[iema_recente['CEG'] == usina].drop(columns=['CEG']).reset_index(drop=True)
        
        # Unificação dos dados de geração, emissão e dados unitários
        if previsao == '1':
            df = pd.merge(geracao, emissao, on=None, how='left').merge(unitarios, on=None, how='cross')
        else:
            df = geracao.merge(unitarios, on=None, how='cross')
        
        
        # == TRANSFORMAÇÃO DAS HORAS (ÍNDICE) EM VALORES CÍCLICOS ==
        # Transformar o índice de horas em valores cíclicos para capturar a natureza periódica dos dados.
        horas = {}
        anos = sorted([os.path.splitext(ano)[0] for ano in set(os.listdir('IEMA/Tabelas')) - {'desktop.ini'}])
        for ano in anos:
            horas[ano] = 8784 if int(ano) % 4 == 0 else 8760
        H = df['Ano'].map(horas)  # Obter o número de horas para cada ano
        theta = 2 * np.pi * df['Índice'] / H  # Calcular o ângulo para cada hora

        df['Cosseno Índice'] = np.cos(theta)
        df['Seno Índice'] = np.sin(theta)


        # == FEATURES DE TRANSIÇÃO ==
        # Features importantes para a rede neural tratar as transições como um processo, e não como um estado fixo.
        
        # Duração da transição
        # Flag de transição (1 se estiver em transição, 0 caso contrário)
        transicao_list = list((df['Categoria de geração'] == 0).astype(int))
        duracao = np.zeros_like(transicao_list, dtype=int)
        contador = 0
        for i in range(len(transicao_list)):
            if transicao_list[i]:
                contador += 1
            else:
                contador = 0
            duracao[i] = contador
        df['Duração da Transição'] = duracao
        
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
        df['Fase da Transição'] = fase
        
        # Definir as colunas categóricas e numéricas para o ColumnTransformer
        categoricas = ['Categoria de geração']
        
        # Adicionar outras colunas numéricas, se necessário
        numericas = ['Geração', 'Duração da Transição', 'Fase da Transição']

        # Configurar o ColumnTransformer para aplicar OneHotEncoder nas colunas categóricas e StandardScaler nas colunas numéricas
        transf = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categoricas),
            ('scaler', StandardScaler(), numericas)],
                                   remainder='passthrough',  # Mantém constant_cols inalteradas
                                   verbose_feature_names_out=False  # Para nomes mais limpos
                                   )
        
        # Separar os dados em treino, validação e teste com base no ano. Remover as colunas que não serão usadas como features.
        x_tr = df[df['Ano'].astype(int) < 2023].reset_index(drop=True).drop(columns=['Delta menos', 'Delta mais', 'Índice', 'Ano'])
        x_val = df[df['Ano'].astype(int) == 2023].reset_index(drop=True).drop(columns=['Delta menos', 'Delta mais', 'Índice', 'Ano'])
        x_te = df[df['Ano'].astype(int) > 2023].reset_index(drop=True).drop(columns=['Delta menos', 'Delta mais', 'Índice', 'Ano'])

        # Aplicar as transformações usando o ColumnTransformer.
        # Apenas os dados de treino são ajustados (fit) para evitar vazamento de dados, 
        # enquanto os dados de validação e teste são transformados (transform) usando os parâmetros ajustados no treino.
        data_tr = transf.fit_transform(x_tr)
        data_val = transf.transform(x_val)
        data_te = transf.transform(x_te)

        # Geração dos dataframes finais (Treino, Validação e Teste) com os nomes das colunas resultantes do ColumnTransformer.
        df_tr = pd.DataFrame(data_tr, columns=transf.get_feature_names_out())
        df_val = pd.DataFrame(data_val, columns=transf.get_feature_names_out())
        df_te = pd.DataFrame(data_te, columns=transf.get_feature_names_out())

        # Dataframe com os dados de normalização para aplicação na equação da rede PGNM
        scaler = transf.named_transformers_['scaler']

        # Posição da coluna "Geração" dentro de numericas
        idx_pg = numericas.index('Geração')

        df_norm = pd.DataFrame({
            'Variável': ['Geração'],
            'Mean': [scaler.mean_[idx_pg]],
            'Std': [scaler.scale_[idx_pg]]
        })
        
        # Salvar os dataframes em formato parquet
        if previsao == '1':
            dir = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Dados')
        else:
            dir = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Dados')

        os.makedirs(os.path.join(dir), exist_ok=True)
        df_tr.to_parquet(os.path.join(dir, 'Treino.parquet'), index=False)
        df_val.to_parquet(os.path.join(dir, 'Validação.parquet'), index=False)
        df_te.to_parquet(os.path.join(dir, 'Teste.parquet'), index=False)
        df_norm.to_parquet(os.path.join(dir, 'Scaler Geração.parquet'), index=False)


def main(previsao):
    # Ler a tabela mais recente do IEMA e filtrar os dados para as usinas presentes na pasta "Usinas"
    iema_recente = pd.read_excel(os.path.join('IEMA/Tabelas', sorted(os.listdir('IEMA/Tabelas'))[-1]))

    colunas_remover = ['Município', 
                    'Geração [GWh]', 
                    'Emissões de Gases [tCO2]', 
                    'Taxa de Emissão [tCO2/GWh]']

    if os.listdir('Usinas')[0][:3] == 'UTE':
        iema_recente = iema_recente[iema_recente[f'CEG'].isin(os.listdir('Usinas'))].reset_index(drop=True)
        colunas_remover.append('Usina')
    else:
        iema_recente = iema_recente[iema_recente[f'Usina'].isin(os.listdir('Usinas'))].reset_index(drop=True)
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
    tratamento_dados(iema_recente, previsao)