import pandas as pd
import os, numpy as np

def compilar_coeficientes(previsao='1', modo='Regressão L2'):
    """
    Compila os coeficientes de todas as usinas para um determinado modo.
    
    Parâmetros:
        previsao: '1' para Minimização, '2' para PGNM
        modo: 'Padrão' ou 'Regressão L2' (apenas para previsao='1')
    """
    resultados = []
    
    for usina in os.listdir('Usinas'):
        # Constrói o caminho do arquivo de coeficientes
        if previsao == '1':
            caminho_coef = os.path.join('Usinas', usina, 'Minimização', modo, 'Emissões', 'Coeficientes.parquet')
        else:
            caminho_coef = os.path.join('Usinas', usina, 'PGNM', 'Emissões', 'Coeficientes.parquet')
        
        if os.path.exists(caminho_coef):
            df_coef = pd.read_parquet(caminho_coef)
            
            # Transforma o DataFrame de long para wide
            coeficientes = df_coef.set_index('Coeficientes')['Valores'].to_dict()
            coeficientes['usina'] = usina
            
            resultados.append(coeficientes)
    
    if not resultados:
        print(f"Nenhum arquivo de coeficientes encontrado para o modo {modo}")
        return None
    
    # Cria DataFrame consolidado
    df_consolidado = pd.DataFrame(resultados)
    
    # Reordena as colunas para uma apresentação melhor
    colunas = ['usina', 'Alpha', 'Beta', 'Gamma', 'Omega', 'Mu', 'MSE']
    df_consolidado = df_consolidado[[col for col in colunas if col in df_consolidado.columns]]
    
    return df_consolidado


def gerar_tabela_resumo_comparativa():
    """
    Gera uma tabela comparativa entre os métodos Padrão e Regressão L2,
    com médias e desvios padrão dos coeficientes.
    """
    # Compila os dados para ambos os métodos
    df_padrao = compilar_coeficientes(previsao='1', modo='Padrão')
    df_l2 = compilar_coeficientes(previsao='1', modo='Regressão L2')
    
    if df_padrao is None or df_l2 is None:
        print("Erro ao carregar os dados")
        return None, None, None
    
    # Calcula estatísticas descritivas
    def calcular_estatisticas(df, nome_metodo):
        estatisticas = []
        for col in ['Alpha', 'Beta', 'Gamma', 'Omega', 'Mu', 'MSE']:
            if col in df.columns:
                # Remove valores nulos ou infinitos
                dados = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(dados) > 0:
                    estatisticas.append({
                        'Método': nome_metodo,
                        'Coeficiente': col,
                        'Média': dados.mean(),
                        'Desvio Padrão': dados.std(),
                        'Mínimo': dados.min(),
                        'Máximo': dados.max(),
                        'Contagem': len(dados)
                    })
        return pd.DataFrame(estatisticas)
    
    # Cria tabelas de estatísticas
    df_estat_padrao = calcular_estatisticas(df_padrao, 'Padrão')
    df_estat_l2 = calcular_estatisticas(df_l2, 'Regressão L2')
    
    # Concatena as duas tabelas
    df_resumo = pd.concat([df_estat_padrao, df_estat_l2], ignore_index=True)
    
    return df_resumo, df_padrao, df_l2


def exportar_tabelas():
    """
    Exporta as tabelas geradas para arquivos CSV e LaTeX.
    """
    import numpy as np
    
    # Cria diretório para os resultados
    os.makedirs('Resultados/Coeficientes', exist_ok=True)
    
    # Gera as tabelas
    df_resumo, df_padrao, df_l2 = gerar_tabela_resumo_comparativa()
    
    if df_resumo is None:
        return
    
    # Exporta dados completos por usina
    df_padrao.to_csv('Resultados/Coeficientes/Coeficientes_Padrao.csv', index=False)
    df_l2.to_csv('Resultados/Coeficientes/Coeficientes_Regressao_L2.csv', index=False)
    
    # Exporta tabela resumo
    df_resumo.to_csv('Resultados/Coeficientes/Resumo_Comparativo.csv', index=False)
    
    # Gera tabela em formato LaTeX
    with open('Resultados/Coeficientes/Tabela_Coeficientes.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparação dos Coeficientes Médios}\n")
        f.write("\\label{tab:coeficientes}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Método} & \\textbf{Coeficiente} & \\textbf{Média} & \\textbf{Desvio} & \\textbf{Min} & \\textbf{Max} \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df_resumo.iterrows():
            f.write(f"{row['Método']} & {row['Coeficiente']} & {row['Média']:.4f} & {row['Desvio Padrão']:.4f} & {row['Mínimo']:.4f} & {row['Máximo']:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print("✅ Tabelas exportadas com sucesso!")
    print(f"  - Resultados/Coeficientes/Coeficientes_Padrao.csv")
    print(f"  - Resultados/Coeficientes/Coeficientes_Regressao_L2.csv")
    print(f"  - Resultados/Coeficientes/Resumo_Comparativo.csv")
    print(f"  - Resultados/Coeficientes/Tabela_Coeficientes.tex")
    
    # Exibe um resumo no terminal
    print("\n📊 RESUMO DOS COEFICIENTES (Regressão L2):")
    print(df_resumo[df_resumo['Método'] == 'Regressão L2'].to_string(index=False))


if __name__ == "__main__":
    exportar_tabelas()