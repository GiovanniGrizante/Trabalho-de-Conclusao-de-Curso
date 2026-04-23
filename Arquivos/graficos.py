import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, time

def configs(msg, intervalo):
    while True:
        os.system('cls')
        resp = int(input(msg))
        if resp in intervalo:
            return resp
        print('Opção inválida!')
        time.sleep(3)


def plotar_geracao_emissao(usina, previsao='1', modo='Padrão'):
    """
    Gera grafico com dois eixos Y: Geracao e Emissao
    Eixo X: Tempo (Indice com marcacao de Ano)
    
    Parâmetros:
        usina: nome da usina
        previsao: '1' para minimização, '2' para PINN
        modo: 'Padrão', 'Regressão L2' ou 'PGNM'
    """
    # Define o caminho base conforme previsao
    if previsao == '1':
        tipo_analise_path = 'Minimização'
        # Para previsao='1', modo pode ser 'Padrão' ou 'Regressão L2'
        caminho_emissao = os.path.join('Usinas', usina, tipo_analise_path, modo, 'Emissões', 'Horárias.parquet')
    else:  # previsao == '2'
        tipo_analise_path = 'PGNM'
        caminho_emissao = os.path.join('Usinas', usina, tipo_analise_path, 'Emissões', 'Horárias.parquet')
    
    # Caminhos dos arquivos
    caminho_geracao = os.path.join('Usinas', usina, 'Dados Externos', 'ONS.parquet')
    
    # Carregar dados
    df_geracao = pd.read_parquet(caminho_geracao)
    df_emissao = pd.read_parquet(caminho_emissao)
    
    # Verificar se as colunas necessárias existem
    col_indice = 'Indice' if 'Indice' in df_geracao.columns else 'Índice'
    col_ano = 'Ano' if 'Ano' in df_geracao.columns else 'ano'
    
    # Unir dados
    df_merged = pd.merge(
        df_geracao, 
        df_emissao, 
        on=[col_indice, col_ano], 
        how='inner'
    )
    
    if len(df_merged) == 0:
        return
    
    # Ordenar por indice
    df_merged = df_merged.sort_values(col_indice)
    
    # Obter anos únicos
    anos_unicos = sorted(df_merged[col_ano].unique())
    
    # Criar pasta para salvar graficos
    if previsao == '1':
        pasta_usina = os.path.join('Usinas', usina, tipo_analise_path, modo, 'Gráficos', 'Geração X Emissão')
    else:
        pasta_usina = os.path.join('Usinas', usina, tipo_analise_path, 'Gráficos', 'Geração X Emissão')
    os.makedirs(pasta_usina, exist_ok=True)
    
    # Gerar um grafico para cada ano
    for ano in anos_unicos:
        df_ano = df_merged[df_merged[col_ano] == ano].copy()
        
        if len(df_ano) == 0:
            continue
        
        # Criar figura com dois eixos Y
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # Eixo Y1: Geracao (esquerda)
        cor_geracao = '#2E86AB'
        ax1.set_xlabel('Hora do Ano')
        ax1.set_ylabel('Geracao (MW)', color=cor_geracao)
        ax1.plot(df_ano[col_indice], df_ano['Geração'], 
                 color=cor_geracao, linewidth=1, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=cor_geracao)
        ax1.grid(True, alpha=0.3)
        
        # Eixo Y2: Emissao (direita)
        ax2 = ax1.twinx()
        cor_emissao = '#E15554'
        ax2.set_ylabel('Emissao (tCO2/h)', color=cor_emissao)
        ax2.plot(df_ano[col_indice], df_ano['Emissão'], 
                 color=cor_emissao, linewidth=1, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=cor_emissao)
        
        # Titulo
        modo_str = f" - {modo}" if previsao == '1' else ""
        plt.title(f'Geração e Emissão - {usina}{modo_str} - {ano}', fontsize=14)
        
        # Legenda manual
        legend_elements = [
            Line2D([0], [0], color=cor_geracao, linewidth=2, label='Geração (MW)'),
            Line2D([0], [0], color=cor_emissao, linewidth=2, label='Emissão (tCO2/h)')
        ]
        ax1.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        # Salvar grafico em SVG
        nome_arquivo = f'{ano}.svg'
        plt.savefig(os.path.join(pasta_usina, nome_arquivo), format='svg', bbox_inches='tight')
        plt.close()


def plotar_gerais(resultados_df, graphs, ext):
    
    # ========== GRÁFICOS ==========
    
    def histograma(resultados_df, graphs, ext, path):
        # Histograma do MAE (escala log)
        if graphs == plt:
            graphs.figure(figsize=(10, 6))
        
        graphs.hist(np.log10(resultados_df['mae'] + 1e-10), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        graphs.axvline(np.log10(resultados_df['mae'].mean()), color='red', 
                linestyle='--', linewidth=2,
                label=f'Média: {resultados_df["mae"].mean():.2f}')
        graphs.axvline(np.log10(resultados_df['mae'].median()), color='green', 
                linestyle='--', linewidth=2,
                label=f'Mediana: {resultados_df["mae"].median():.2f}')
        graphs.xlabel('log10(MAE) [tCO2/h]')
        graphs.ylabel('Número de Usinas')
        graphs.title('Distribuição do MAE (escala log)')
        graphs.legend()
        graphs.grid(True, alpha=0.3)
        
        if graphs == plt:
            os.makedirs(os.path.join(path, f'{ext[1:].upper()}'), exist_ok=True)
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Histograma MAE{ext}'), format='svg', bbox_inches='tight')
            plt.close()
        
    def dispersao(resultados_df, graphs, ext, path):
        # 2. Gráfico de dispersão
        df_sorted = resultados_df.sort_values('mae', ascending=False).reset_index(drop=True)
        x_pos = np.arange(len(df_sorted))
        colors = np.log10(df_sorted['mae'].values + 1e-10)

        if graphs == plt:
            graphs.figure(figsize=(12, 6))

        scatter = graphs.scatter(x_pos, df_sorted['mae'].values, 
                            c=colors, cmap='viridis', alpha=0.7, s=50)

        graphs.axhline(y=resultados_df['mae'].median(), color='red', 
                    linestyle='--', linewidth=2, 
                    label=f'Mediana: {resultados_df["mae"].median():.2f}')
        graphs.axhline(y=resultados_df['mae'].mean(), color='blue', 
                    linestyle='--', linewidth=2, 
                    label=f'Média: {resultados_df["mae"].mean():.2f}')

        graphs.xlabel('Usinas (ordenadas por MAE decrescente)')
        graphs.ylabel('MAE (tCO2/h)')
        graphs.title('Dispersão do MAE por Usina')
        graphs.yscale('log')
        graphs.legend(loc='upper right')
        graphs.grid(True, alpha=0.3, axis='y')
        graphs.colorbar(scatter, label='log10(MAE)')
        graphs.tight_layout()
        
        if graphs == plt:
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Dispersão{ext}'), format='svg', bbox_inches='tight')
            plt.close()
    
    def mae_mse(resultados_df, graphs, ext, path):
        # MAE xs MSE
        if graphs == plt:
            graphs.figure(figsize=(10, 6))
        
        graphs.scatter(resultados_df['mae'], resultados_df['mse'], 
                alpha=0.6, c='steelblue', s=40)
        graphs.xlabel('MAE')
        graphs.ylabel('MSE')
        graphs.title('MAE x MSE')
        graphs.xscale('log')
        graphs.yscale('log')
        graphs.grid(True, alpha=0.3)
        graphs.tight_layout()
        
        if graphs == plt:
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'MAE x MSE{ext}'), format='svg', bbox_inches='tight')
            plt.close()
    
    def melhores(resultados_df, graphs, ext, path):
        # Top 10 melhores usinas
        top10_melhores = resultados_df.nsmallest(10, 'mae')
        cores_melhores = plt.cm.YlOrRd(1 - top10_melhores['mae'] / top10_melhores['mae'].max())

        if graphs == plt:
            graphs.figure(figsize=(10, 6))
        
        graphs.barh(range(len(top10_melhores)), top10_melhores['mae'].values, 
                color=cores_melhores, edgecolor='black')
        
        # Tratamento diferente para plt e axes
        if graphs == plt:
            graphs.yticks(range(len(top10_melhores)), top10_melhores['usina'].values, fontsize=8)
            graphs.gca().invert_yaxis()
        else:
            graphs.set_yticks(range(len(top10_melhores)))
            graphs.set_yticklabels(top10_melhores['usina'].values, fontsize=8)
            graphs.invert_yaxis()
        
        graphs.xlabel('MAE (tCO2/h)')
        graphs.title('Top 10 Melhores Usinas (menor MAE)')
        graphs.grid(True, alpha=0.3, axis='x')
        graphs.tight_layout()
        
        if graphs == plt:
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'10 Melhores Desempenhos{ext}'), format='svg', bbox_inches='tight')
            plt.close()
    
    def piores(resultados_df, graphs, ext, path):
        top10_piores = resultados_df.nlargest(10, 'mae')
        cores_piores = plt.cm.Reds(top10_piores['mae'] / top10_piores['mae'].max())
        
        if graphs == plt:
            graphs.figure(figsize=(10, 6))
        
        graphs.barh(range(len(top10_piores)), top10_piores['mae'].values, 
                        color=cores_piores, edgecolor='black')
        
        # Tratamento diferente para plt e axes
        if graphs == plt:
            graphs.yticks(range(len(top10_piores)), top10_piores['usina'].values, fontsize=8)
            graphs.gca().invert_yaxis()
            graphs.xlabel('MAE (tCO2/h)')
            graphs.title('Top 10 Piores Usinas (maior MAE)')
        else:
            graphs.set_yticks(range(len(top10_piores)))
            graphs.set_yticklabels(top10_piores['usina'].values, fontsize=8)
            graphs.invert_yaxis()
            graphs.set_xlabel('MAE (tCO2/h)')
            graphs.set_title('Top 10 Piores Usinas (maior MAE)')
        
        graphs.grid(True, alpha=0.3, axis='x')
        graphs.tight_layout()
        
        if graphs == plt:
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'10 Piores Desempenhos{ext}'), format='svg', bbox_inches='tight')
            plt.close()
    
    def resumo(resultados_df, graphs, ext, path):
        # Calcula as estatísticas
        q1 = resultados_df['mae'].quantile(0.25)
        q3 = resultados_df['mae'].quantile(0.75)
        iqr = q3 - q1
        
        # Organiza os dados em categorias
        dados_tabela = [
            ['TOTAL DE USINAS', f'{len(resultados_df)}', ''],
            ['', '', ''],
            ['MAE (tCO2/h)', '', ''],
            ['  Média', f'{resultados_df["mae"].mean():.2f}', ''],
            ['  Mediana', f'{resultados_df["mae"].median():.2f}', ''],
            ['  Desvio Padrão', f'{resultados_df["mae"].std():.2f}', ''],
            ['  Mínimo', f'{resultados_df["mae"].min():.6f}', ''],
            ['  Máximo', f'{resultados_df["mae"].max():.2f}', ''],
            ['  1º Quartil', f'{q1:.2f}', ''],
            ['  3º Quartil', f'{q3:.2f}', ''],
            ['  IQR', f'{iqr:.2f}', ''],
            ['', '', ''],
            ['DESEMPENHO', '', ''],
            ['  MAE < 10', f'{(resultados_df["mae"] < 10).sum()}', f'{(resultados_df["mae"] < 10).sum() / len(resultados_df) * 100:.1f}%'],
            ['  MAE < 100', f'{(resultados_df["mae"] < 100).sum()}', f'{(resultados_df["mae"] < 100).sum() / len(resultados_df) * 100:.1f}%'],
            ['', '', ''],
            ['PROBLEMÁTICAS', '', ''],
            ['  MAE > 1000', f'{(resultados_df["mae"] > 1000).sum()}', f'{(resultados_df["mae"] > 1000).sum() / len(resultados_df) * 100:.1f}%'],
            ['  MAE > 10000', f'{(resultados_df["mae"] > 10000).sum()}', f'{(resultados_df["mae"] > 10000).sum() / len(resultados_df) * 100:.1f}%']
        ]
        
        # Verifica se é gráfico individual (plt) ou subplot (axes)
        if graphs == plt:
            # Para gráficos individuais, cria uma figura separada
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            # Cria a tabela
            table = ax.table(cellText=dados_tabela,
                            colLabels=['Métrica', 'Valor', '% do Total'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.5, 0.2, 0.2])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Estiliza as células
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Cabeçalho
                    cell.set_facecolor('#2E4053')
                    cell.set_text_props(weight='bold', color='white')
                elif i in [1, 12, 16]:  # Linhas de separação
                    cell.set_facecolor('#F0F0F0')
                elif dados_tabela[i-1][0] == 'TOTAL DE USINAS':
                    cell.set_facecolor('#D5E8D4')
                    cell.set_text_props(weight='bold')
                elif dados_tabela[i-1][0] == 'MAE (tCO2/h)':
                    cell.set_facecolor('#DAE3F3')
                    cell.set_text_props(weight='bold')
                elif dados_tabela[i-1][0] == 'DESEMPENHO':
                    cell.set_facecolor('#D5E8D4')
                    cell.set_text_props(weight='bold')
                elif dados_tabela[i-1][0] == 'PROBLEMÁTICAS':
                    cell.set_facecolor('#F8CECC')
                    cell.set_text_props(weight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Resumo Estatístico{ext}'), 
                    format='svg', bbox_inches='tight')
            plt.close()
            
        else:
            # Para subplots, usa o axes existente (sem criar nova figura)
            graphs.axis('off')  # Desliga os eixos
            
            # Cria a tabela no axes existente
            table = graphs.table(cellText=dados_tabela,
                                colLabels=['Métrica', 'Valor', '% do Total'],
                                cellLoc='left',
                                loc='center',
                                colWidths=[0.5, 0.2, 0.2])
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)  # Fonte um pouco menor para subplot
            table.scale(1.1, 1.3)  # Escala ligeiramente menor
            
            # Estiliza as células
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Cabeçalho
                    cell.set_facecolor('#2E4053')
                    cell.set_text_props(weight='bold', color='white')
                elif i in [1, 12, 16]:  # Linhas de separação
                    cell.set_facecolor('#F0F0F0')
                elif dados_tabela[i-1][0].startswith('📊'):
                    cell.set_facecolor('#D5E8D4')
                    cell.set_text_props(weight='bold')
                elif dados_tabela[i-1][0].startswith('📈'):
                    cell.set_facecolor('#DAE3F3')
                    cell.set_text_props(weight='bold')
                elif dados_tabela[i-1][0].startswith('✅'):
                    cell.set_facecolor('#D5E8D4')
                    cell.set_text_props(weight='bold')
                elif dados_tabela[i-1][0].startswith('⚠️'):
                    cell.set_facecolor('#F8CECC')
                    cell.set_text_props(weight='bold')
    
    # ========== EXECUÇÃO PRINCIPAL ==========
    
    if graphs == 1:
        os.makedirs(os.path.join('Resultados', 'Gráficos', 'Individuais'), exist_ok=True)
        path = os.path.join('Resultados', 'Gráficos', 'Individuais')
        
        # Redefine graphs para ser o plt
        graphs = plt
        
        histograma(resultados_df, graphs, ext, path)
        dispersao(resultados_df, graphs, ext, path)
        mae_mse(resultados_df, graphs, ext, path)
        melhores(resultados_df, graphs, ext, path)
        piores(resultados_df, graphs, ext, path)
        resumo(resultados_df, graphs, ext, path)
        
    elif graphs == 2:
        os.makedirs(os.path.join('Resultados', 'Gráficos', 'Unificados'), exist_ok=True)
        path = os.path.join('Resultados', 'Gráficos', 'Unificados')
        
        # Cria a figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Chama cada função com o axes correspondente
        histograma(resultados_df, axes[0, 0], ext, path)
        dispersao(resultados_df, axes[0, 1], ext, path)
        mae_mse(resultados_df, axes[0, 2], ext, path)
        melhores(resultados_df, axes[1, 0], ext, path)
        piores(resultados_df, axes[1, 1], ext, path)
        resumo(resultados_df, axes[1, 2], ext, path)
        
        # Ajusta e salva a figura completa
        plt.suptitle('Resultados do Modelo - Análise de Emissões (tCO2/h)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'Gerais{ext}'), format='svg', bbox_inches='tight')
        plt.close()


def plotar_instabilidade_geral(resultados_df, graphs, ext):
    
    def distribuicao(df_analise, graphs, ext, path):
        if graphs == plt:
            graphs.figure(figsize=(10, 6))
            
        # Distribuicao da razao MSE/MAE2
        graphs.hist(df_analise['razao'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        graphs.axvline(x=3, color='orange', linestyle='--', linewidth=1.5, label='Limiar moderado (3)')
        graphs.axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='Limiar alto (5)')
        graphs.axvline(x=10, color='darkred', linestyle='--', linewidth=1.5, label='Limiar extremo (10)')
        
        if graphs == plt:
            graphs.xlabel('Razão MSE/MAE²')
            graphs.ylabel('Número de Usinas')
            graphs.title('Distribuição da Instabilidade')
            graphs.legend()
            graphs.grid(True, alpha=0.3)
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Distribuição de Instabilidade{ext}'), format='svg', bbox_inches='tight')
            plt.close()
            
        else:
            graphs.set_xlabel('Razão MSE/MAE²')
            graphs.set_ylabel('Número de Usinas')
            graphs.set_title('Distribuição da Instabilidade')
            graphs.legend()
            graphs.grid(True, alpha=0.3)
    
    def categorias(df_analise, graphs, ext, path):
        if graphs == plt:
            graphs.figure(figsize=(10, 6))
            
        # Boxplot por categoria de instabilidade
        categorias = ['Estável', 'Moderada', 'Alta', 'Extrema']
        dados_box = []
        for cat in categorias:
            valores = df_analise[df_analise['instabilidade'] == cat]['razao'].round(3).values
            if len(valores) > 0:
                dados_box.append(valores)
            else:
                dados_box.append([])
        
        bp = graphs.boxplot(dados_box, tick_labels=categorias, patch_artist=True)
        cores_box = ['green', 'orange', 'red', 'darkred']
        for patch, cor in zip(bp['boxes'], cores_box):
            if patch is not None:
                patch.set_facecolor(cor)
                patch.set_alpha(0.5)
        
        if graphs == plt:
            graphs.ylabel('Razao MSE/MAE²')
            graphs.title('Instabilidade por Categoria')
            graphs.grid(True, alpha=0.3, axis='y')
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Instabilidade por Categoria{ext}'), format='svg', bbox_inches='tight')
            plt.close()
        
        else:
            graphs.set_ylabel('Razao MSE/MAE²')
            graphs.set_title('Instabilidade por Categoria')
            graphs.grid(True, alpha=0.3, axis='y')
    
    def dispersao(df_analise, graphs, ext, path):
        if graphs == plt:
            graphs.figure(figsize=(10, 6))
            
        # Dispersao MAE vs Razao com cores
        cores_instab = {'Estável': 'green', 'Moderada': 'orange', 'Alta': 'red', 'Extrema': 'darkred'}
        for cat, cor in cores_instab.items():
            subset = df_analise[df_analise['instabilidade'] == cat]
            if len(subset) > 0:
                graphs.scatter(subset['mae'], subset['razao'], c=cor, label=cat, alpha=0.6, s=40)
        graphs.axhline(y=3, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        graphs.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.7)
        graphs.axhline(y=10, color='darkred', linestyle='--', linewidth=1, alpha=0.7)
        
        if graphs == plt:
            graphs.xlabel('MAE (tCO2/h)')
            graphs.ylabel('Razao MSE/MAE²')
            graphs.title('MAE vs Instabilidade')
            graphs.xscale('log')
            graphs.yscale('log')
            graphs.legend()
            graphs.grid(True, alpha=0.3)
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'MAE x Instabilidade{ext}'), format='svg', bbox_inches='tight')
            plt.close()
        
        else:
            graphs.set_xlabel('MAE (tCO2/h)')
            graphs.set_ylabel('Razao MSE/MAE²')
            graphs.set_title('MAE vs Instabilidade')
            graphs.set_xscale('log')
            graphs.set_yscale('log')
            graphs.legend()
            graphs.grid(True, alpha=0.3)
            
    df_analise = resultados_df.copy()
    df_analise['razao'] = round(df_analise['mse'] / (df_analise['mae'] ** 2),3)
    df_analise['instabilidade'] = 'Estável'
    
    df_analise.loc[df_analise['razao'] > 3, 'instabilidade'] = 'Moderada'
    df_analise.loc[df_analise['razao'] > 5, 'instabilidade'] = 'Alta'
    df_analise.loc[df_analise['razao'] > 10, 'instabilidade'] = 'Extrema'
    
    if graphs == 1:
        path = os.path.join('Resultados', 'Gráficos', 'Individuais')
        
        # Redefine graphs para ser o plt
        graphs = plt
        
        distribuicao(df_analise, graphs, ext, path)
        categorias(df_analise, graphs, ext, path)
        dispersao(df_analise, graphs, ext, path)
    
    elif graphs == 2:
        path = os.path.join('Resultados', 'Gráficos', 'Unificados')
        
        # Cria a figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Chama cada função com o axes correspondente
        distribuicao(df_analise, axes[0], ext, path)
        categorias(df_analise, axes[1], ext, path)
        dispersao(df_analise, axes[2], ext, path)
        
        # Ajusta e salva a figura completa
        plt.suptitle('Análise Geral da Instabilidade dos Modelos', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'Gerais{ext}'), format='svg', bbox_inches='tight')
        plt.close()
    
    return df_analise


def plotar_analise_usina(resultados_df, usina_alvo, previsao='1', modo='Padrão'):
    """
    Analise detalhada de instabilidade para uma usina especifica
    Salva o grafico em: Usinas/[usina]/[tipo_analise]/[modo]/Graficos/Instabilidade.png
    """
    if usina_alvo not in resultados_df['usina'].values:
        return
    
    # Filtra apenas os dados do modo específico
    df_filtrado = resultados_df[(resultados_df['usina'] == usina_alvo) & 
                                 (resultados_df['modo_analise'] == modo)]
    
    if len(df_filtrado) == 0:
        return
    
    row = df_filtrado.iloc[0]
    mae = row['mae']
    mse = row['mse']
    razao = mse / (mae ** 2) if mae > 0 else 0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Indicadores de instabilidade
    axes[0].axis('off')
    
    # Classificacao da usina
    if razao > 10:
        status = "EXTREMA"
        cor_status = "darkred"
        recomendacao = "Modelo muito instável - Não recomendado para uso"
    elif razao > 5:
        status = "ALTA"
        cor_status = "red"
        recomendacao = "Modelo com outliers - Validar antes de usar"
    elif razao > 3:
        status = "MODERADA"
        cor_status = "orange"
        recomendacao = "Modelo com alguma instabilidade - Monitorar"
    else:
        status = "BAIXA"
        cor_status = "green"
        recomendacao = "Modelo estável - Confiável para uso"
    
    texto = f"""ANÁLISE DE INSTABILIDADE

MÉTRICAS GLOBAIS:
  MAE: {mae:.6f}
  MSE: {mse:.4f}
  MAE²: {(mae**2):.8f}
  MSE/MAE²: {razao:.2f}

CLASSIFICAÇÃO: {status}
  (Razão > 10: Extrema | >5: Alta | >3: Moderada | <=3: Baixa)

SIGNIFICADO:
  - MSE/MAE² > 10: Modelo extremamente instável (outliers muito grandes)
  - MSE/MAE² > 5:  Há erros muito grandes (outliers)
  - MSE/MAE² 3 a 5: Instabilidade moderada
  - MSE/MAE² < 3:  Erros bem distribuídos (modelo estável)

RECOMENDAÇÃO:
  {recomendacao}
"""
    axes[0].text(0.05, 0.95, texto, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Grafico comparativo com outras usinas (filtrado pelo mesmo modo)
    df_comp = resultados_df[resultados_df['modo_analise'] == modo].copy()
    df_comp['razao'] = df_comp['mse'] / (df_comp['mae'] ** 2)
    
    cores = ['steelblue' if u != usina_alvo else cor_status for u in df_comp['usina']]
    sizes = [30 if u != usina_alvo else 100 for u in df_comp['usina']]
    
    axes[1].scatter(df_comp['mae'], df_comp['razao'], c=cores, s=sizes, alpha=0.6)
    axes[1].axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Limiar instabilidade (5)')
    axes[1].axhline(y=3, color='orange', linestyle='--', linewidth=1, label='Limiar atenção (3)')
    axes[1].set_xlabel('MAE (tCO2/h)')
    axes[1].set_ylabel('Razão MSE/MAE²')
    axes[1].set_title(f'Posição da Usina no Contexto Geral ({modo})')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    modo_str = f" - {modo}" if previsao == '1' else ""
    plt.suptitle(f'Análise de Instabilidade - {usina_alvo}{modo_str}', fontsize=14)
    plt.tight_layout()
    
    # Salvar na pasta da usina
    if previsao == '1':
        tipo_analise_path = 'Minimização'
        pasta_usina = os.path.join('Usinas', usina_alvo, tipo_analise_path, modo, 'Gráficos')
    else:
        tipo_analise_path = 'PGNM'
        pasta_usina = os.path.join('Usinas', usina_alvo, tipo_analise_path, 'Gráficos')
    os.makedirs(pasta_usina, exist_ok=True)
    plt.savefig(os.path.join(pasta_usina, 'Instabilidade.svg'), format='svg', bbox_inches='tight')
    plt.close()


def plotar_erro_temporal(usina, previsao='1', modo='Padrão'):
    """
    Analisa erro temporal de uma usina específica
    Salva gráficos na pasta da usina: Usinas/[usina]/[tipo_analise]/[modo]/Graficos/
    """
    # Define o caminho base conforme previsao
    if previsao == '1':
        tipo_analise_path = 'Minimização'
        caminho_base = os.path.join('Usinas', usina, tipo_analise_path, modo, 'Rede Neural')
    else:
        tipo_analise_path = 'PGNM'
        caminho_base = os.path.join('Usinas', usina, tipo_analise_path, 'Rede Neural')
    
    # Caminhos
    caminho_predicoes = os.path.join(caminho_base, 'Historicos', 'Predições.parquet')
    caminho_dados = os.path.join(caminho_base, 'Dados', 'Teste.parquet')
    
    if not os.path.exists(caminho_predicoes):
        return None
    
    # Carregar predições
    df_pred = pd.read_parquet(caminho_predicoes)
    
    # Carregar dados originais
    df_dados = pd.read_parquet(caminho_dados) if os.path.exists(caminho_dados) else None
    
    y_true = df_pred['y_true'].values
    y_pred = df_pred['y_pred'].values
    
    # Calcular erros
    erro = y_pred - y_true
    erro_abs = np.abs(erro)
    erro_quad = erro ** 2
    
    # Métricas globais
    mae_global = np.mean(erro_abs)
    mse_global = np.mean(erro_quad)
    razao_global = mse_global / (mae_global ** 2) if mae_global > 0 else 0
    
    # Calcular MSE/MAE² por janela deslizante (168 horas = 7 dias)
    window = 168
    razao_local = []
    for i in range(len(erro) - window):
        mae_local = np.mean(np.abs(erro[i:i+window]))
        mse_local = np.mean(erro[i:i+window]**2)
        if mae_local > 0:
            razao_local.append(mse_local / (mae_local**2))
        else:
            razao_local.append(0)
    
    # Calcular estatísticas dos maiores erros
    top_10_indices = np.argsort(erro_abs)[-10:]  # 10 maiores erros
    top_10_erros = erro_abs[top_10_indices]
    
    # Criar figura com múltiplos subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Série temporal: Real vs Previsto (primeiras 500 horas)
    n_mostrar = min(500, len(y_true))
    axes[0, 0].plot(y_true[:n_mostrar], label='Real', alpha=0.7, linewidth=1, color='blue')
    axes[0, 0].plot(y_pred[:n_mostrar], label='Previsto', alpha=0.7, linewidth=1, color='red', linestyle='--')
    axes[0, 0].set_ylabel('Emissões (tCO2/h)')
    axes[0, 0].set_title(f'Real vs Previsto (primeiras {n_mostrar} horas)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Erro ao longo do tempo
    axes[0, 1].plot(erro[:n_mostrar], color='purple', alpha=0.7, linewidth=0.8)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 1].fill_between(range(n_mostrar), 0, erro[:n_mostrar], 
                            where=(erro[:n_mostrar] > 0), color='red', alpha=0.3, label='Erro positivo')
    axes[0, 1].fill_between(range(n_mostrar), erro[:n_mostrar], 0, 
                            where=(erro[:n_mostrar] < 0), color='blue', alpha=0.3, label='Erro negativo')
    axes[0, 1].set_ylabel('Erro (tCO2/h)')
    axes[0, 1].set_title('Erro ao longo do tempo')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Erro Absoluto com limiares
    limiar_95 = np.percentile(erro_abs, 95)
    limiar_99 = np.percentile(erro_abs, 99)
    
    axes[1, 0].plot(erro_abs[:n_mostrar], color='orange', alpha=0.7, linewidth=0.8)
    axes[1, 0].axhline(y=limiar_95, color='red', linestyle='--', 
                       label=f'95º percentil: {limiar_95:.2f}')
    axes[1, 0].axhline(y=limiar_99, color='darkred', linestyle='--', 
                       label=f'99º percentil: {limiar_99:.2f}')
    axes[1, 0].set_ylabel('Erro Absoluto (tCO2/h)')
    axes[1, 0].set_title('Erro Absoluto e Outliers')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. MSE/MAE² por janela deslizante (INSTABILIDADE LOCAL)
    axes[1, 1].plot(razao_local, color='darkgreen', alpha=0.7, linewidth=0.8)
    axes[1, 1].axhline(y=3, color='green', linestyle='--', linewidth=1, label='Estável (<3)')
    axes[1, 1].axhline(y=5, color='orange', linestyle='--', linewidth=1, label='Alerta (5)')
    axes[1, 1].axhline(y=10, color='red', linestyle='--', linewidth=1, label='Extremo (>10)')
    axes[1, 1].set_ylabel('MSE/MAE²')
    axes[1, 1].set_xlabel('Tempo (horas)')
    axes[1, 1].set_title(f'Instabilidade Local (janela {window}h)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Dispersão: Erro vs Valor Real
    axes[2, 0].scatter(y_true, erro_abs, alpha=0.3, s=10, c='steelblue')
    axes[2, 0].axhline(y=limiar_95, color='red', linestyle='--', label=f'95º percentil')
    axes[2, 0].set_xlabel('Valor Real (tCO2/h)')
    axes[2, 0].set_ylabel('Erro Absoluto')
    axes[2, 0].set_title('Erro vs Valor Real')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Histograma dos erros
    axes[2, 1].hist(erro, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[2, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[2, 1].axvline(x=np.mean(erro), color='red', linestyle='--', 
                       label=f'Média: {np.mean(erro):.2f}')
    axes[2, 1].set_xlabel('Erro (tCO2/h)')
    axes[2, 1].set_ylabel('Frequência')
    axes[2, 1].set_title('Distribuição dos Erros')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Título principal com métricas
    classificacao = "Extrema" if razao_global > 10 else ("Alta" if razao_global > 5 else ("Moderada" if razao_global > 3 else "Estável"))
    
    modo_str = f" - {modo}" if previsao == '1' else ""
    plt.suptitle(f'{usina}{modo_str} | MAE: {mae_global:.2f} | MSE/MAE²: {razao_global:.2f} | Classificação: {classificacao}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico na pasta da usina
    if previsao == '1':
        pasta_usina = os.path.join('Usinas', usina, tipo_analise_path, modo, 'Gráficos')
    else:
        pasta_usina = os.path.join('Usinas', usina, tipo_analise_path, 'Gráficos')
    os.makedirs(pasta_usina, exist_ok=True)
    plt.savefig(os.path.join(pasta_usina, 'Erro Temporal.svg'), format='svg', bbox_inches='tight')
    plt.close()


def plotar_comparativo_emissoes_calculadas(usina):
    """
    Gera um gráfico comparativo entre os métodos 'Padrão' e 'Regressão L2'
    usando os dados de emissões calculadas (Horárias.parquet)
    Mostra as emissões ao longo do tempo para ambos os métodos
    Salva o gráfico em: Usinas/[usina]/Minimização/Gráficos/Comparativo_Emissoes_Padrao_vs_L2.svg
    """
    
    # Caminhos para os arquivos de emissões calculadas
    caminhos = {
        'Padrão': os.path.join('Usinas', usina, 'Minimização', 'Padrão', 'Emissões', 'Horárias.parquet'),
        'Regressão L2': os.path.join('Usinas', usina, 'Minimização', 'Regressão L2', 'Emissões', 'Horárias.parquet')
    }
    
    # Carregar dados de cada método
    dados = {}
    anos_disponiveis = {}
    
    for metodo, caminho in caminhos.items():
        if os.path.exists(caminho):
            df = pd.read_parquet(caminho)
            # Verificar se tem a coluna 'Emissão'
            if 'Emissão' in df.columns:
                dados[metodo] = df['Emissão'].values
                # Tentar extrair informações de ano se disponível
                if 'Ano' in df.columns:
                    anos_disponiveis[metodo] = df['Ano'].values
            else:
                print(f"  ⚠️ {metodo}: coluna 'Emissão' não encontrada")
    
    if len(dados) < 2:
        print(f"  ⚠️ Usina {usina}: dados insuficientes para comparação")
        return None
    
    # Garantir que os arrays tenham o mesmo tamanho
    min_len = min(len(dados['Padrão']), len(dados['Regressão L2']))
    y_padrao = dados['Padrão'][:min_len]
    y_l2 = dados['Regressão L2'][:min_len]
    
    # Calcular diferença
    diferenca = y_padrao - y_l2
    
    # Calcular estatísticas
    mae_diferenca = np.mean(np.abs(diferenca))
    correlacao = np.corrcoef(y_padrao, y_l2)[0, 1]
    
    # Criar figura com 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Série temporal: Emissões Calculadas - Comparação
    n_mostrar = min(1000, min_len)
    axes[0].plot(y_padrao[:n_mostrar], color='#2E86AB', linewidth=1, alpha=0.8, label='Padrão')
    axes[0].plot(y_l2[:n_mostrar], color='#E15554', linewidth=1, alpha=0.8, label='Regressão L2')
    axes[0].set_ylabel('Emissões (tCO2/h)')
    axes[0].set_title(f'Comparação de Emissões Calculadas - {usina} (primeiras {n_mostrar} horas)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Diferença entre os métodos
    axes[1].plot(diferenca[:n_mostrar], color='purple', linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].fill_between(range(n_mostrar), 0, diferenca[:n_mostrar], 
                         where=(diferenca[:n_mostrar] > 0), color='green', alpha=0.3, 
                         label='Padrão > L2')
    axes[1].fill_between(range(n_mostrar), diferenca[:n_mostrar], 0, 
                         where=(diferenca[:n_mostrar] < 0), color='red', alpha=0.3, 
                         label='L2 > Padrão')
    axes[1].set_ylabel('Diferença (Padrão - L2)')
    axes[1].set_title('Diferença entre Métodos (positivo = Padrão superestima)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Dispersão: Padrão vs L2
    axes[2].scatter(y_padrao, y_l2, alpha=0.3, s=10, c='steelblue')
    
    # Linha de referência y=x (igualdade perfeita)
    min_val = min(y_padrao.min(), y_l2.min())
    max_val = max(y_padrao.max(), y_l2.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, 
                 label='Igualdade perfeita')
    
    axes[2].set_xlabel('Emissões - Método Padrão (tCO2/h)')
    axes[2].set_ylabel('Emissões - Regressão L2 (tCO2/h)')
    axes[2].set_title(f'Dispersão: Padrão vs L2 (correlação: {correlacao:.4f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Estatísticas gerais
    mae_padrao = np.mean(np.abs(y_padrao))
    mae_l2 = np.mean(np.abs(y_l2))
    
    plt.suptitle(f'Comparativo de Emissões Calculadas: Padrão vs Regressão L2\n'
                 f'MAE Padrão: {mae_padrao:.2f} | MAE L2: {mae_l2:.2f} | '
                 f'Diferença média absoluta: {mae_diferenca:.2f} | Correlação: {correlacao:.4f}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    pasta_destino = os.path.join('Usinas', usina, 'Minimização', 'Gráficos')
    os.makedirs(pasta_destino, exist_ok=True)
    plt.savefig(os.path.join(pasta_destino, 'Comparativo_Emissoes_Padrao_vs_L2.svg'), 
                format='svg', bbox_inches='tight')
    plt.close()
    
    # Retornar estatísticas
    return {
        'usina': usina,
        'mae_padrao': mae_padrao,
        'mae_l2': mae_l2,
        'diferenca_media': np.mean(diferenca),
        'diferenca_mediana': np.median(diferenca),
        'diferenca_std': np.std(diferenca),
        'correlacao': correlacao,
        'mae_diferenca': mae_diferenca
    }


def plotar_comparativo_todas_usinas_emissoes():
    """
    Gera gráficos comparativos de emissões calculadas para todas as usinas 
    que possuem ambos os métodos (Padrão e Regressão L2)
    """
    resultados_comparacao = []
    
    for usina in os.listdir('Usinas'):
        # Verificar se a usina tem ambos os métodos
        caminho_padrao = os.path.join('Usinas', usina, 'Minimização', 'Padrão', 'Emissões', 'Horárias.parquet')
        caminho_l2 = os.path.join('Usinas', usina, 'Minimização', 'Regressão L2', 'Emissões', 'Horárias.parquet')
        
        if os.path.exists(caminho_padrao) and os.path.exists(caminho_l2):
            resultado = plotar_comparativo_emissoes_calculadas(usina)
            if resultado:
                resultados_comparacao.append(resultado)
    
    # Salvar resumo da comparação
    if resultados_comparacao:
        df_resumo = pd.DataFrame(resultados_comparacao)
        pasta_destino = os.path.join('Resultados', 'previsao_1', 'Comparativos_Emissoes')
        os.makedirs(pasta_destino, exist_ok=True)
        df_resumo.to_csv(os.path.join(pasta_destino, 'Resumo_Comparativo_Emissoes_Padrao_vs_L2.csv'), 
                        index=False)
        
        # Criar gráfico de barras comparativo (MAE)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico 1: MAE Padrão vs L2
        x = np.arange(len(df_resumo['usina']))
        width = 0.35
        
        axes[0].bar(x - width/2, df_resumo['mae_padrao'], width, label='Padrão', 
                    color='#2E86AB', alpha=0.7)
        axes[0].bar(x + width/2, df_resumo['mae_l2'], width, label='Regressão L2', 
                    color='#E15554', alpha=0.7)
        axes[0].set_xlabel('Usinas')
        axes[0].set_ylabel('MAE (tCO2/h)')
        axes[0].set_title('MAE das Emissões: Padrão vs Regressão L2')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_resumo['usina'], rotation=45, ha='right', fontsize=8)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Gráfico 2: Correlação entre métodos
        cores = ['green' if c > 0.95 else 'orange' if c > 0.9 else 'red' for c in df_resumo['correlacao']]
        axes[1].barh(df_resumo['usina'], df_resumo['correlacao'], color=cores, alpha=0.7)
        axes[1].axvline(x=0.95, color='green', linestyle='--', linewidth=1, label='Excelente (0.95)')
        axes[1].axvline(x=0.90, color='orange', linestyle='--', linewidth=1, label='Bom (0.90)')
        axes[1].set_xlabel('Correlação')
        axes[1].set_title('Correlação entre Métodos Padrão e L2')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Comparação Geral: Padrão vs Regressão L2', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_destino, 'Resumo_MAE_Correlacao_Padrao_vs_L2.svg'), 
                    format='svg', bbox_inches='tight')
        plt.close()
        
    return resultados_comparacao


def plotar_comparativo_regressao_RN(usina):
    """
    Gera um gráfico comparativo entre os métodos 'Padrão' e 'Regressão L2'
    Mostra as emissões previstas por ambos os métodos ao longo do tempo
    Salva o gráfico em: Usinas/[usina]/Minimização/Gráficos/Comparativo_Padrao_vs_L2.svg
    """
    
    # Caminhos para os dois métodos
    caminhos = {
        'Padrão': os.path.join('Usinas', usina, 'Minimização', 'Padrão', 'Rede Neural', 'Historicos', 'Predições.parquet'),
        'Regressão L2': os.path.join('Usinas', usina, 'Minimização', 'Regressão L2', 'Rede Neural', 'Historicos', 'Predições.parquet')
    }
    
    # Carregar dados de cada método
    dados = {}
    for metodo, caminho in caminhos.items():
        if os.path.exists(caminho):
            df = pd.read_parquet(caminho)
            # Verificar se tem a coluna correta
            if 'y_pred' in df.columns:
                dados[metodo] = df['y_pred'].values
            elif 'Emissões Previstas' in df.columns:
                dados[metodo] = df['Emissões Previstas'].values
            else:
                # Tentar encontrar qualquer coluna numérica
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    dados[metodo] = df[num_cols[0]].values
                else:
                    print(f"  ⚠️ {metodo}: nenhuma coluna numérica encontrada")
    
    if len(dados) < 2:
        print(f"  ⚠️ Usina {usina}: dados insuficientes para comparação (Padrão: {'✓' if 'Padrão' in dados else '✗'}, L2: {'✓' if 'Regressão L2' in dados else '✗'})")
        return
    
    # Garantir que os arrays tenham o mesmo tamanho
    min_len = min(len(dados['Padrão']), len(dados['Regressão L2']))
    y_padrao = dados['Padrão'][:min_len]
    y_l2 = dados['Regressão L2'][:min_len]
    
    # Calcular diferença
    diferenca = y_padrao - y_l2
    
    # Criar figura com 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Série temporal: Emissões Previstas - Método Padrão
    axes[0].plot(y_padrao[:1000], color='#2E86AB', linewidth=1, alpha=0.8, label='Padrão')
    axes[0].plot(y_l2[:1000], color='#E15554', linewidth=1, alpha=0.8, label='Regressão L2')
    axes[0].set_ylabel('Emissões (tCO2/h)')
    axes[0].set_title(f'Comparação de Previsões - {usina} (primeiras 1000 horas)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Diferença entre os métodos
    axes[1].plot(diferenca[:1000], color='purple', linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].fill_between(range(1000), 0, diferenca[:1000], 
                         where=(diferenca[:1000] > 0), color='green', alpha=0.3, label='Padrão > L2')
    axes[1].fill_between(range(1000), diferenca[:1000], 0, 
                         where=(diferenca[:1000] < 0), color='red', alpha=0.3, label='L2 > Padrão')
    axes[1].set_ylabel('Diferença (Padrão - L2)')
    axes[1].set_title('Diferença entre Métodos (positivo = Padrão superestima)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Estatísticas da diferença (histograma)
    axes[2].hist(diferenca, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Diferença zero')
    axes[2].axvline(x=np.mean(diferenca), color='red', linestyle='--', 
                    label=f'Média: {np.mean(diferenca):.2f}')
    axes[2].axvline(x=np.median(diferenca), color='green', linestyle='--', 
                    label=f'Mediana: {np.median(diferenca):.2f}')
    axes[2].set_xlabel('Diferença (Padrão - L2)')
    axes[2].set_ylabel('Frequência')
    axes[2].set_title('Distribuição das Diferenças')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Estatísticas gerais
    mae_padrao = np.mean(np.abs(y_padrao))
    mae_l2 = np.mean(np.abs(y_l2))
    mae_diferenca = np.mean(np.abs(diferenca))
    
    plt.suptitle(f'Comparativo: Padrão vs Regressão L2\n'
                 f'MAE Padrão: {mae_padrao:.2f} | MAE L2: {mae_l2:.2f} | '
                 f'Diferença média absoluta: {mae_diferenca:.2f}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    pasta_destino = os.path.join('Usinas', usina, 'Minimização', 'Gráficos')
    os.makedirs(pasta_destino, exist_ok=True)
    plt.savefig(os.path.join(pasta_destino, 'Comparativo_Padrao_vs_L2.svg'), 
                format='svg', bbox_inches='tight')
    plt.close()
    
    # Retornar estatísticas para possível uso
    return {
        'usina': usina,
        'mae_padrao': mae_padrao,
        'mae_l2': mae_l2,
        'diferenca_media': np.mean(diferenca),
        'diferenca_mediana': np.median(diferenca),
        'diferenca_std': np.std(diferenca)
    }


def plotar_comparativo_todas_usinas_RN():
    """
    Gera gráficos comparativos para todas as usinas que possuem ambos os métodos
    """
    resultados_comparacao = []
    
    for usina in os.listdir('Usinas'):
        # Verificar se a usina tem ambos os métodos
        caminho_padrao = os.path.join('Usinas', usina, 'Minimização', 'Padrão', 'Rede Neural', 'Historicos', 'Predições.parquet')
        caminho_l2 = os.path.join('Usinas', usina, 'Minimização', 'Regressão L2', 'Rede Neural', 'Historicos', 'Predições.parquet')
        
        if os.path.exists(caminho_padrao) and os.path.exists(caminho_l2):
            resultado = plotar_comparativo_regressao_RN(usina)
            if resultado:
                resultados_comparacao.append(resultado)
    
    # Salvar resumo da comparação
    if resultados_comparacao:
        df_resumo = pd.DataFrame(resultados_comparacao)
        pasta_destino = os.path.join('Resultados', 'previsao_1', 'Comparativos')
        os.makedirs(pasta_destino, exist_ok=True)
        df_resumo.to_csv(os.path.join(pasta_destino, 'Resumo_Comparativo_Padrao_vs_L2.csv'), index=False)
        
        # Criar gráfico de barras comparativo
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_resumo['usina']))
        width = 0.35
        
        ax.bar(x - width/2, df_resumo['mae_padrao'], width, label='Padrão', color='#2E86AB', alpha=0.7)
        ax.bar(x + width/2, df_resumo['mae_l2'], width, label='Regressão L2', color='#E15554', alpha=0.7)
        
        ax.set_xlabel('Usinas')
        ax.set_ylabel('MAE (tCO2/h)')
        ax.set_title('Comparação de MAE: Padrão vs Regressão L2')
        ax.set_xticks(x)
        ax.set_xticklabels(df_resumo['usina'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_destino, 'Resumo_MAE_Padrao_vs_L2.svg'), 
                    format='svg', bbox_inches='tight')
        plt.close()
    
    return resultados_comparacao


def main(previsao):
    graphs = configs('Gráficos individuais ou unificados?\n(1 - Individuais | 2 - Unificados): ', [1, 2])
    
    ext = '.svg'  # Forçando SVG
    
    # =========================================================
    # DEFINIÇÃO DAS PASTAS BASEADO EM 'previsao'
    # =========================================================
    if previsao == '1':  # Modo Minimização
        modos_analise = ['Padrão', 'Regressão L2']
        tipo_analise_path = 'Minimização'
    else:  # previsao == '2' - Modo Híbrido (PGNM)
        modos_analise = ['PGNM']
        tipo_analise_path = 'PGNM'

    resultados = []
    
    # Loop principal para cada usina e modo
    for usina in os.listdir('Usinas'):
        for modo in modos_analise:
            caminho_teste = os.path.join('Usinas', usina, tipo_analise_path, modo, 'Rede Neural', 'Históricos', 'Teste.parquet')
            
            if os.path.exists(caminho_teste):
                df_teste = pd.read_parquet(caminho_teste)
                df_teste['modo_analise'] = modo
                df_teste['previsao'] = previsao
                resultados.append(df_teste)
    
    if not resultados:
        print(f'Sem dados disponíveis para geração dos gráficos com a previsão {previsao}.')
        return
    
    df_resultados = pd.concat(resultados, ignore_index=True)
    
    # --- Geração dos DataFrames e Gráficos Gerais ---
    save_path = os.path.join('Resultados', f'previsao_{previsao}')
    os.makedirs(os.path.join(save_path, 'Erros'), exist_ok=True)
    df_resultados.to_parquet(os.path.join(save_path, 'Erros', 'Por Usina.parquet'))
    
    # Gráficos gerais (precisam ser adaptados para usar save_path)
    plotar_gerais(df_resultados, graphs, ext)
    plotar_instabilidade_geral(df_resultados, graphs, ext)
    
    # Gerar gráficos individuais por usina e modo
    for usina in df_resultados['usina'].unique():
        for modo in modos_analise:
            # Verificar se existe dados para esta combinação
            df_filtrado = df_resultados[(df_resultados['usina'] == usina) & (df_resultados['modo_analise'] == modo)]
            if len(df_filtrado) > 0:
                plotar_analise_usina(df_resultados, usina, previsao, modo)
                plotar_erro_temporal(usina, previsao, modo)
                plotar_geracao_emissao(usina, previsao, modo)
                
    # =========================================================
    # COMPARATIVO PADRÃO VS REGRESSÃO L2 (apenas para previsao='1')
    # =========================================================
    if previsao == '1':
        plotar_comparativo_todas_usinas_RN()
        plotar_comparativo_todas_usinas_emissoes()


if __name__ == "__main__":
    # Exemplo de chamada
    main(previsao='1')