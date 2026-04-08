import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, time

def plotar_geracao_emissao(usina):
    """
    Gera grafico com dois eixos Y: Geracao e Emissao
    Eixo X: Tempo (Indice com marcacao de Ano)
    """
    # Caminhos dos arquivos
    caminho_geracao = os.path.join('Usinas', usina, 'Dados Externos', 'ONS.parquet')
    caminho_emissao = os.path.join('Usinas', usina, 'Minimização', 'Emissões', 'Horárias.parquet')
    
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
    pasta_usina = os.path.join('Usinas', usina, 'Minimização', 'Gráficos', 'Geração X Emissão')
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
        plt.title(f'Geração e Emissão - {usina} - {ano}', fontsize=14)
        
        # Legenda manual
        legend_elements = [
            Line2D([0], [0], color=cor_geracao, linewidth=2, label='Geração (MW)'),
            Line2D([0], [0], color=cor_emissao, linewidth=2, label='Emissão (tCO2/h)')
        ]
        ax1.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        # Salvar grafico
        nome_arquivo = f'{ano}.png'
        plt.savefig(os.path.join(pasta_usina, nome_arquivo), dpi=150, bbox_inches='tight')
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
            os.makedirs(os.path.join(path, f'{ext[1:].upper()}'),exist_ok=True)
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Histograma MAE{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Dispersão{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'MAE x MSE{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'10 Melhores Desempenhos{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'10 Piores Desempenhos{ext}'), dpi=300, bbox_inches='tight')
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
            
            # ax.set_title('Resumo Estatístico - Análise de Emissões (tCO2/h)', 
            #             fontsize=14, weight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Resumo Estatístico{ext}'), 
                    dpi=300, bbox_inches='tight')
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
            
            # Título menor para subplot
            #graphs.set_title('Resumo Estatístico', fontsize=10, weight='bold')
    
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
        plt.savefig(os.path.join(path, f'Gerais{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Distribuição de Instabilidade{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'Instabilidade por Categoria{ext}'), dpi=300, bbox_inches='tight')
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
            graphs.savefig(os.path.join(path, f'{ext[1:].upper()}', f'MAE x Instabilidade{ext}'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(path, f'Gerais{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return df_analise

def plotar_analise_usina(resultados_df, usina_alvo):
    """
    Analise detalhada de instabilidade para uma usina especifica
    Salva o grafico em: Usinas/[usina]/Minimização/Graficos/Instabilidade.png
    """
    if usina_alvo not in resultados_df['usina'].values:
        return
    
    row = resultados_df[resultados_df['usina'] == usina_alvo].iloc[0]
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
    
    # 2. Grafico comparativo com outras usinas
    df_comp = resultados_df.copy()
    df_comp['razao'] = df_comp['mse'] / (df_comp['mae'] ** 2)
    
    cores = ['steelblue' if u != usina_alvo else cor_status for u in df_comp['usina']]
    sizes = [30 if u != usina_alvo else 100 for u in df_comp['usina']]
    
    axes[1].scatter(df_comp['mae'], df_comp['razao'], c=cores, s=sizes, alpha=0.6)
    axes[1].axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Limiar instabilidade (5)')
    axes[1].axhline(y=3, color='orange', linestyle='--', linewidth=1, label='Limiar atenção (3)')
    axes[1].set_xlabel('MAE (tCO2/h)')
    axes[1].set_ylabel('Razão MSE/MAE²')
    axes[1].set_title(f'Posição da Usina no Contexto Geral')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Análise de Instabilidade - {usina_alvo}', fontsize=14)
    plt.tight_layout()
    
    # Salvar na pasta da usina em Usinas/[usina]/Graficos/
    pasta_usina = os.path.join('Usinas', usina_alvo, 'Minimização', 'Gráficos')
    os.makedirs(pasta_usina, exist_ok=True)
    plt.savefig(os.path.join(pasta_usina, 'Instabilidade.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plotar_erro_temporal(usina):
    """
    Analisa erro temporal de uma usina específica
    Salva gráficos na pasta da usina: Usinas/[usina]/Graficos/
    """
    # Caminhos
    caminho_predicoes = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Historicos', 'Predições.parquet')
    caminho_dados = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Dados', 'Teste.parquet')
    
    if not os.path.exists(caminho_predicoes):
        return None
    
    # Carregar predições
    df_pred = pd.read_parquet(caminho_predicoes)
    
    # Carregar dados originais para obter informações adicionais (categoria, geração, etc.)
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
    
    plt.suptitle(f'{usina} | MAE: {mae_global:.2f} | MSE/MAE²: {razao_global:.2f} | Classificação: {classificacao}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico na pasta da usina
    pasta_usina = os.path.join('Usinas', usina, 'Minimização', 'Gráficos')
    os.makedirs(pasta_usina, exist_ok=True)
    plt.savefig(os.path.join(pasta_usina, 'Erro Temporal.png'), dpi=150, bbox_inches='tight')
    plt.close()

def configs(msg, intervalo):
    while True:
        os.system('cls')
        resp = int(input(msg))
        if resp in intervalo:
            return resp
        print('Opção inválida!')
        time.sleep(3)

def main():
    graphs = configs('Gráficos individuais ou unificados?\n(1 - Individuais | 2 - Unificados): ', [1, 2])
    
    ext = configs('Qual o tipo de gráfico?\n(1 - PNG | 2 - SVG): ', [1, 2])
    ext_dict = {1: '.png', 2: '.svg'}
    ext = ext_dict[ext]
    
    resultados = []
    
    for usina in os.listdir('Usinas'):
        caminho_teste = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Históricos', 'Teste.parquet')
        
        if os.path.exists(caminho_teste):
            df_teste = pd.read_parquet(caminho_teste)
            resultados.append(df_teste)
    
    if not resultados:
        print('Sem dados disponíveis para geração dos gráficos.')
        return
    
    df_resultados = pd.concat(resultados, ignore_index=True)
    
    df_geral = pd.DataFrame([
        ['Total de usinas', len(df_resultados)],
        ['MAE medio', f"{df_resultados['mae'].mean():.2f} ± {df_resultados['mae'].std():.2f}"],
        ['MAE mediano', f"{df_resultados['mae'].median():.2f}"],
        ['MAE minimo', f"{df_resultados['mae'].min():.6f}"],
        ['MAE maximo', f"{df_resultados['mae'].max():.2f}"],
        ['MSE medio', f"{df_resultados['mse'].mean():.2f}"]
    ], columns=['Estatistica', 'Valor'])
    
    df_geral['Valor'] = df_geral['Valor'].astype(str)
    

    os.makedirs(os.path.join('Resultados', 'Erros'), exist_ok=True)
    df_resultados.to_parquet(os.path.join('Resultados', 'Erros', 'Por Usina.parquet'))
    df_geral.to_parquet(os.path.join('Resultados', 'Erros', 'Gerais.parquet'))
    
    plotar_gerais(df_resultados, graphs, ext)
    plotar_instabilidade_geral(df_resultados, graphs, ext)
    
    # Gerar grafico para cada usina
    for usina in df_resultados['usina'].values:
        plotar_analise_usina(df_resultados, usina)
        plotar_geracao_emissao(usina)