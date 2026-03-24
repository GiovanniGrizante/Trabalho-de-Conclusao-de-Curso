import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def plotar_geracao_emissao(usina):
    """
    Gera grafico com dois eixos Y: Geracao e Emissao
    Eixo X: Tempo (Indice com marcacao de Ano)
    """
    # Caminhos dos arquivos
    caminho_geracao = os.path.join('Dados Tratados', usina, 'Dados Externos', 'ONS.parquet')
    caminho_emissao = os.path.join('Dados Tratados', usina, 'Emissões Sintéticas', 'Horárias.parquet')
    
    if not os.path.exists(caminho_geracao) or not os.path.exists(caminho_emissao):
        return
    
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
    pasta_usina = os.path.join('Dados Tratados', usina, 'Gráficos', 'Geração X Emissão')
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

def plotar_gerais(resultados_df):
    """
    Plota resultados em tCO2/h (escala real) com grafico de dispersao
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Histograma do MAE (escala log)
    axes[0, 0].hist(np.log10(resultados_df['mae'] + 1e-10), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(np.log10(resultados_df['mae'].mean()), color='red', 
                       linestyle='--', linewidth=2,
                       label=f'Media: {resultados_df["mae"].mean():.2f}')
    axes[0, 0].axvline(np.log10(resultados_df['mae'].median()), color='green', 
                       linestyle='--', linewidth=2,
                       label=f'Mediana: {resultados_df["mae"].median():.2f}')
    axes[0, 0].set_xlabel('log10(MAE) [tCO2/h]')
    axes[0, 0].set_ylabel('Numero de Usinas')
    axes[0, 0].set_title('Distribuicao do MAE (escala log)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Grafico de dispersao
    df_sorted = resultados_df.sort_values('mae', ascending=False).reset_index(drop=True)
    x_pos = np.arange(len(df_sorted))
    colors = np.log10(df_sorted['mae'].values + 1e-10)
    scatter = axes[0, 1].scatter(x_pos, df_sorted['mae'].values, 
                                 c=colors, cmap='viridis', alpha=0.7, s=50)
    axes[0, 1].axhline(y=resultados_df['mae'].median(), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mediana: {resultados_df["mae"].median():.2f}')
    axes[0, 1].axhline(y=resultados_df['mae'].mean(), color='blue', 
                       linestyle='--', linewidth=2, 
                       label=f'Media: {resultados_df["mae"].mean():.2f}')
    axes[0, 1].set_xlabel('Usinas (ordenadas por MAE decrescente)')
    axes[0, 1].set_ylabel('MAE (tCO2/h)')
    axes[0, 1].set_title('Dispersao do MAE por Usina')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    plt.colorbar(scatter, ax=axes[0, 1], label='log10(MAE)')
    
    # 3. MAE vs MSE
    axes[0, 2].scatter(resultados_df['mae'], resultados_df['mse'], 
                       alpha=0.6, c='steelblue', s=40)
    axes[0, 2].set_xlabel('MAE')
    axes[0, 2].set_ylabel('MSE')
    axes[0, 2].set_title('MAE vs MSE')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Top 10 melhores usinas
    top10_melhores = resultados_df.nsmallest(10, 'mae')
    cores_melhores = plt.cm.YlOrRd(1 - top10_melhores['mae'] / top10_melhores['mae'].max())
    axes[1, 0].barh(range(len(top10_melhores)), top10_melhores['mae'].values, 
                    color=cores_melhores, edgecolor='black')
    axes[1, 0].set_yticks(range(len(top10_melhores)))
    axes[1, 0].set_yticklabels(top10_melhores['usina'].values, fontsize=8)
    axes[1, 0].set_xlabel('MAE (tCO2/h)')
    axes[1, 0].set_title('Top 10 Melhores Usinas (menor MAE)')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. Top 10 piores usinas
    top10_piores = resultados_df.nlargest(10, 'mae')
    cores_piores = plt.cm.Reds(top10_piores['mae'] / top10_piores['mae'].max())
    axes[1, 1].barh(range(len(top10_piores)), top10_piores['mae'].values, 
                    color=cores_piores, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top10_piores)))
    axes[1, 1].set_yticklabels(top10_piores['usina'].values, fontsize=8)
    axes[1, 1].set_xlabel('MAE (tCO2/h)')
    axes[1, 1].set_title('Top 10 Piores Usinas (maior MAE)')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Resumo estatistico
    axes[1, 2].axis('off')
    q1 = resultados_df['mae'].quantile(0.25)
    q3 = resultados_df['mae'].quantile(0.75)
    iqr = q3 - q1
    
    texto = f"""RESUMO ESTATÍSTICO (tCO2/h)

Total de Usinas: {len(resultados_df)}

MAE:
  Média: {resultados_df['mae'].mean():.2f}
  Mediana: {resultados_df['mae'].median():.2f}
  Desvio Padrão: {resultados_df['mae'].std():.2f}
  
  Mínimo: {resultados_df['mae'].min():.6f}
  Máximo: {resultados_df['mae'].max():.2f}
  
  1º Quartil: {q1:.2f}
  3º Quartil: {q3:.2f}
  IQR: {iqr:.2f}

DESEMPENHO:
  MAE < 10: {(resultados_df['mae'] < 10).sum()} usinas
  MAE < 100: {(resultados_df['mae'] < 100).sum()} usinas

PROBLEMÁTICAS:
  MAE > 1000: {(resultados_df['mae'] > 1000).sum()} usinas
  MAE > 10000: {(resultados_df['mae'] > 10000).sum()} usinas
"""
    axes[1, 2].text(0.05, 0.95, texto, transform=axes[1, 2].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Resultados do Modelo - Análise de Emissões (tCO2/h)', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.join('Resultados', 'Gráficos'), exist_ok=True)
    plt.savefig(os.path.join('Resultados', 'Gráficos', 'Gerais.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plotar_instabilidade_geral(resultados_df):
    """
    Analise geral de instabilidade para todas as usinas
    """
    df_analise = resultados_df.copy()
    df_analise['razao'] = df_analise['mse'] / (df_analise['mae'] ** 2)
    df_analise['instabilidade'] = 'Estável'
    df_analise.loc[df_analise['razao'] > 3, 'instabilidade'] = 'Moderada'
    df_analise.loc[df_analise['razao'] > 5, 'instabilidade'] = 'Alta'
    df_analise.loc[df_analise['razao'] > 10, 'instabilidade'] = 'Extrema'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Distribuicao da razao MSE/MAE2
    axes[0].hist(df_analise['razao'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=3, color='orange', linestyle='--', linewidth=1.5, label='Limiar moderado (3)')
    axes[0].axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='Limiar alto (5)')
    axes[0].axvline(x=10, color='darkred', linestyle='--', linewidth=1.5, label='Limiar extremo (10)')
    axes[0].set_xlabel('Razão MSE/MAE²')
    axes[0].set_ylabel('Número de Usinas')
    axes[0].set_title('Distribuição da Instabilidade')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Boxplot por categoria de instabilidade
    categorias = ['Estável', 'Moderada', 'Alta', 'Extrema']
    dados_box = []
    for cat in categorias:
        valores = df_analise[df_analise['instabilidade'] == cat]['razao'].values
        if len(valores) > 0:
            dados_box.append(valores)
        else:
            dados_box.append([])
    
    bp = axes[1].boxplot(dados_box, tick_labels=categorias, patch_artist=True)
    cores_box = ['green', 'orange', 'red', 'darkred']
    for patch, cor in zip(bp['boxes'], cores_box):
        if patch is not None:
            patch.set_facecolor(cor)
            patch.set_alpha(0.5)
    axes[1].set_ylabel('Razao MSE/MAE²')
    axes[1].set_title('Instabilidade por Categoria')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Dispersao MAE vs Razao com cores
    cores_instab = {'Estável': 'green', 'Moderada': 'orange', 'Alta': 'red', 'Extrema': 'darkred'}
    for cat, cor in cores_instab.items():
        subset = df_analise[df_analise['instabilidade'] == cat]
        if len(subset) > 0:
            axes[2].scatter(subset['mae'], subset['razao'], c=cor, label=cat, alpha=0.6, s=40)
    axes[2].axhline(y=3, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].axhline(y=10, color='darkred', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].set_xlabel('MAE (tCO2/h)')
    axes[2].set_ylabel('Razao MSE/MAE²')
    axes[2].set_title('MAE vs Instabilidade')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Análise Geral de Instabilidade dos Modelos', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.join('Resultados', 'Gráficos'), exist_ok=True)
    plt.savefig(os.path.join('Resultados', 'Gráficos', 'Instabilidade Geral.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Salvar dados de instabilidade
    os.makedirs(os.path.join('Resultados', 'Erros'), exist_ok=True)
    df_analise.to_parquet(os.path.join('Resultados', 'Erros', 'Por Usina.parquet'))
    
    return df_analise

def plotar_analise_usina(resultados_df, usina_alvo):
    """
    Analise detalhada de instabilidade para uma usina especifica
    Salva o grafico em: Dados Tratados/[usina]/Graficos/Instabilidade.png
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
    
    # Salvar na pasta da usina em Dados Tratados/[usina]/Graficos/
    pasta_usina = os.path.join('Dados Tratados', usina_alvo, 'Gráficos')
    os.makedirs(pasta_usina, exist_ok=True)
    plt.savefig(os.path.join(pasta_usina, 'Instabilidade.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    resultados = []
    
    for usina in os.listdir('Dados Tratados'):
        caminho_teste = os.path.join('Dados Tratados', usina, 'Rede Neural', 'Históricos', 'Teste.parquet')
        
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
    
    plotar_gerais(df_resultados)
    plotar_instabilidade_geral(df_resultados)
    
    # Gerar grafico para cada usina
    for usina in df_resultados['usina'].values:
        plotar_analise_usina(df_resultados, usina)
        plotar_geracao_emissao(usina)