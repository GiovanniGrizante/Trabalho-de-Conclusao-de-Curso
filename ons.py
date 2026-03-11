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

def categorizar_geracao(usina, geracao, limiar=15):
    categorias = []
    grupos = []
    proximo_id = 1
    
    for i, valor in enumerate(geracao):
        if valor <= 20:
            valor = 0
            
        if i == 0:
            lim_inf = valor * (1 - limiar/100)
            lim_sup = valor * (1 + limiar/100)
            grupos.append({
                'id': proximo_id,
                'media': valor,
                'limite_inferior': lim_inf,
                'limite_superior': lim_sup,
                'valores': [valor]
            })
            categorias.append(proximo_id)
            proximo_id += 1
        else:
            grupos_validos = []
            for grupo in grupos:
                if grupo['limite_inferior'] <= valor <= grupo['limite_superior']:
                    if grupo['media'] != 0:
                        distancia = abs(valor - grupo['media']) / grupo['media'] * 100
                    else:
                        distancia = abs(valor) * 100
                    grupos_validos.append((grupo, distancia))
            
            if grupos_validos:
                grupos_validos.sort(key=lambda x: x[1])
                grupo_escolhido = grupos_validos[0][0]
                grupo_escolhido['valores'].append(valor)
                nova_media = sum(grupo_escolhido['valores']) / len(grupo_escolhido['valores'])
                grupo_escolhido['media'] = nova_media
                grupo_escolhido['limite_inferior'] = nova_media * (1 - limiar/100)
                grupo_escolhido['limite_superior'] = nova_media * (1 + limiar/100)
                categorias.append(grupo_escolhido['id'])
            else:
                lim_inf = valor * (1 - limiar/100)
                lim_sup = valor * (1 + limiar/100)
                grupos.append({
                    'id': proximo_id,
                    'media': valor,
                    'limite_inferior': lim_inf,
                    'limite_superior': lim_sup,
                    'valores': [valor]
                })
                categorias.append(proximo_id)
                proximo_id += 1
    
    # Mesclar grupos sobrepostos
    while True:
        houve_mesclagem = False
        grupos_ordenados = sorted(grupos, key=lambda g: g['media'])
        novos_grupos = []
        i = 0
        
        while i < len(grupos_ordenados):
            grupo_atual = grupos_ordenados[i].copy()
            j = i + 1
            
            while j < len(grupos_ordenados):
                proximo_grupo = grupos_ordenados[j]
                
                if (grupo_atual['limite_inferior'] <= proximo_grupo['limite_superior'] and 
                    proximo_grupo['limite_inferior'] <= grupo_atual['limite_superior']):
                    
                    grupo_atual['valores'].extend(proximo_grupo['valores'])
                    nova_media = sum(grupo_atual['valores']) / len(grupo_atual['valores'])
                    grupo_atual['media'] = nova_media
                    grupo_atual['limite_inferior'] = nova_media * (1 - limiar/100)
                    grupo_atual['limite_superior'] = nova_media * (1 + limiar/100)
                    grupo_atual['id'] = min(grupo_atual['id'], proximo_grupo['id'])
                    houve_mesclagem = True
                    j += 1
                else:
                    break
            
            novos_grupos.append(grupo_atual)
            i = j
        
        grupos = novos_grupos
        if not houve_mesclagem:
            break
    
    # Renomeia IDs sequencialmente
    for idx, grupo in enumerate(sorted(grupos, key=lambda g: g['media']), 1):
        grupo['id'] = idx
    
    # Reclassifica todos os valores
    novas_categorias = []
    for valor in geracao:
        if valor <= 20:
            valor = 0
            
        grupo_encontrado = None
        for grupo in grupos:
            if grupo['limite_inferior'] <= valor <= grupo['limite_superior']:
                grupo_encontrado = grupo
                break
        
        if grupo_encontrado:
            novas_categorias.append(grupo_encontrado['id'])
        else:
            grupo_mais_proximo = min(grupos, key=lambda g: abs(g['media'] - valor))
            novas_categorias.append(grupo_mais_proximo['id'])
    
    # Segunda passagem: identificar valores de transição com janela de 3 anteriores e 3 posteriores
    categorias_finais = novas_categorias.copy()
    n = len(geracao)
    
    for i in range(n):
        # Pular se já for categoria 0
        if categorias_finais[i] == 0:
            continue
            
        # Coleta categorias dos 3 anteriores (disponíveis)
        categorias_anteriores = []
        for j in range(max(0, i-3), i):
            categorias_anteriores.append(novas_categorias[j])
        
        # Coleta categorias dos 3 posteriores (disponíveis)
        categorias_posteriores = []
        for j in range(i+1, min(n, i+4)):
            categorias_posteriores.append(novas_categorias[j])
        
        # Se não tem elementos suficientes para comparação, pula
        if len(categorias_anteriores) == 0 or len(categorias_posteriores) == 0:
            continue
        
        # Verifica se a categoria atual é diferente da maioria dos anteriores E da maioria dos posteriores
        cat_atual = novas_categorias[i]
        
        # Conta quantos anteriores são diferentes
        diferentes_anteriores = sum(1 for cat in categorias_anteriores if cat != cat_atual)
        # Conta quantos posteriores são diferentes
        diferentes_posteriores = sum(1 for cat in categorias_posteriores if cat != cat_atual)
        
        # Se a maioria dos anteriores E a maioria dos posteriores são diferentes, é transição
        if diferentes_anteriores > len(categorias_anteriores) / 2 and diferentes_posteriores > len(categorias_posteriores) / 2:
            # Verificação adicional: o valor é realmente diferente em magnitude?
            valor_atual = geracao[i] if geracao[i] <= 20 else geracao[i]
            
            # Calcula média dos anteriores (ignorando zeros)
            valores_anteriores = [geracao[j] for j in range(max(0, i-3), i) if geracao[j] > 20 or geracao[j] == 0]
            if valores_anteriores:
                media_anterior = sum(valores_anteriores) / len(valores_anteriores)
            else:
                media_anterior = valor_atual
            
            # Calcula média dos posteriores (ignorando zeros)
            valores_posteriores = [geracao[j] for j in range(i+1, min(n, i+4)) if geracao[j] > 20 or geracao[j] == 0]
            if valores_posteriores:
                media_posterior = sum(valores_posteriores) / len(valores_posteriores)
            else:
                media_posterior = valor_atual
            
            # Verifica se o valor atual é significativamente diferente das médias
            diff_anterior = abs(valor_atual - media_anterior) / max(abs(media_anterior), 1) * 100
            diff_posterior = abs(valor_atual - media_posterior) / max(abs(media_posterior), 1) * 100
            
            if diff_anterior > limiar and diff_posterior > limiar:
                categorias_finais[i] = 0
    
    
    dir = os.path.join('Dados Tratados', usina, 'Informações de Geração')
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    # Gerar arquivo CSV com informações dos grupos
    dados_grupos = []
    for grupo in sorted(grupos, key=lambda g: g['media']):
        qtd = sum(1 for i, c in enumerate(categorias_finais) if c == grupo['id'])
        dados_grupos.append({
            'Grupo': grupo['id'], 
            'Média': round(grupo['media'], 2), 
            'Limite Inferior': round(grupo['limite_inferior'], 2), 
            'Limite Superior': round(grupo['limite_superior'], 2), 
            'Registros': qtd
        })
        
    pd.DataFrame(dados_grupos).to_parquet(os.path.join(dir, 'Grupos de Geração.parquet'), index=False)
    
    
    # Criar lista com os dados de transição
    dados_transicao = []
    for i in range(n):
        if categorias_finais[i] == 0:
            dados_transicao.append({
                'Índice': i, 
                'Geração': round(geracao[i], 3)
            })
            
    pd.DataFrame(dados_transicao).to_parquet(os.path.join(dir, 'Transições de Geração.parquet'), index=False)
                
    return categorias_finais

# Este código acessa os dados da ONS armazenados em arquivos PARQUET, organiza as informações e separa os dados por usina térmica, 
# considerando apenas aquelas que possuem dados de emissão na IEMA para todos os anos estudados.

ons = {}    # Variável para armazenar os dados das usinas térmicas
ons_total = []  # Variável para armazenar todos os dados lidos

# Acessar o drive e armazenar os dados
for ano in anos:
    dir_ano = os.path.join("ONS", ano)

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
            dir_delta = os.path.join("ONS", ano)

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
        
        # Criar a coluna 'Categoria de geração' usando a função de categorização
        geracao = ons[usina]['Geração'].tolist()
        ons[usina]['Categoria de geração'] = categorizar_geracao(usina, geracao)

        # Reorganizar as colunas
        ons[usina] = ons[usina][['Ano', 'Índice', 'Geração','Categoria de geração', 'Delta menos', 'Delta mais']]