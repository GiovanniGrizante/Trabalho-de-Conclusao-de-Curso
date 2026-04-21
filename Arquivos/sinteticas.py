import pandas as pd
import os
import numpy as np, shutil

def main():
    
    # Geração do dict de horas por ano
    horas = {}
    anos =  sorted([os.path.splitext(ano)[0] for ano in set(os.listdir('IEMA/Tabelas')) - {'desktop.ini'}])
    for ano in anos:
        horas[ano] = 8784 if int(ano) % 4 == 0 else 8760
    
    for usina in os.listdir('Usinas'):
        # Geração dos dados de emissão horária por usina
        PG = pd.read_parquet(os.path.join('Usinas', usina, 'Dados Externos', 'ONS.parquet'))['Geração']
        pastas = ['Padrão', 'Regressão L2']

        # Calcular as emissões sintéticas para os casos de minimização (Padrão e Regressão L2)
        for pasta in pastas:
            arq = os.path.join('Usinas', usina, 'Minimização', pasta, 'Emissões', 'Coeficientes.parquet')
            
            if os.path.isfile(arq):
                tab = pd.read_parquet(arq)

                alpha = tab['Valores'].loc[tab['Coeficientes'] == 'Alpha'].item()
                beta = tab['Valores'].loc[tab['Coeficientes'] == 'Beta'].item()
                gamma = tab['Valores'].loc[tab['Coeficientes'] == 'Gamma'].item()
                omega = tab['Valores'].loc[tab['Coeficientes'] == 'Omega'].item()
                mu = tab['Valores'].loc[tab['Coeficientes'] == 'Mu'].item()

                # Cálculo dos valores de emissão sintético conforme equação do artigo
                emissao = []
                for k in range(len(PG)):
                    emissao.append(round((alpha * (PG[k]/100)**2 + beta * (PG[k]/100) + gamma) + omega * np.exp(mu * (PG[k]/100)), 3))
                    
                horas_expandidas = []
                anos_expandidos = []
                for ano in anos:
                    start_idx = 0
                    horas_expandidas.extend(range(start_idx, horas[ano]))
                    anos_expandidos.extend([ano] * horas[ano])

                # Dados para a planilha a ser gerada
                df = pd.DataFrame({
                    'Ano': anos_expandidos,
                    'Índice': horas_expandidas,
                    'Emissão': emissao
                })
                
                df.to_parquet(os.path.join('Usinas', usina, 'Minimização', pasta, 'Emissões', 'Horárias.parquet'), index=False)
                df = df.groupby('Ano')['Emissão'].sum().reset_index()
                
                # Cálculo de emissões anuais e respectivos K_i
                emi_anuais = df['Emissão'].tolist()
                iema = pd.read_parquet(os.path.join('Usinas', usina, 'Dados Externos', 'IEMA.parquet'))['Emissões'].tolist()
                
                df['K_i'] = list(map(lambda x, y: round(x / y, 3), emi_anuais, iema))
                df.to_parquet(os.path.join('Usinas', usina, 'Minimização', pasta, 'Emissões', 'Anuais.parquet'), index=False)
            
            else:
                os.makedirs('Usinas Problemáticas', exist_ok=True)
                try:
                    shutil.move(f'Usinas\\{usina}', 'Usinas Problemáticas')
                    break
                except shutil.Error:
                    shutil.rmtree(f'Usinas\\{usina}')
                    break