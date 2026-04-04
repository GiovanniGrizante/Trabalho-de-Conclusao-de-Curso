import pandas as pd
import os
import numpy as np, shutil

def main():
    
    # Geração do dict de horas por ano
    horas = {}
    anos =  sorted([os.path.splitext(ano)[0] for ano in set(os.listdir('IEMA/Tabelas')) - {'desktop.ini'}])
    for ano in anos:
        horas[ano] = 8784 if int(ano) % 4 == 0 else 8760
    
    
    for usina in os.listdir('Dados Tratados'):
        horas_expandidas = []
        anos_expandidos = []

        # Geração dos dados de emissão horária por usina
        PG = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Dados Externos', 'ONS.parquet'))['Geração']
        emissao = []

        if os.path.isfile(os.path.join('Dados Tratados', usina, 'Emissões Sintéticas', 'Coeficientes.parquet')):
            tab = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Emissões Sintéticas', 'Coeficientes.parquet'))

            alpha = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Alpha'].item()
            beta = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Beta'].item()
            gamma = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Gamma'].item()
            omega = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Omega'].item()
            mu = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Mu'].item()

            # Cálculo dos valores de emissão sintético conforme equação do artigo
            for k in range(len(PG)):
                emissao.append(round((alpha * (PG[k]/100)**2 + beta * (PG[k]/100) + gamma) + omega * np.exp(mu * (PG[k]/100)), 3))
                
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
            
            dir = os.path.join('Dados Tratados', usina)
            df.to_parquet(f'{dir}\\Emissões Sintéticas\\Horárias.parquet', index=False)
            df = df.groupby('Ano')['Emissão'].sum().reset_index()
            
            # Cálculo de emissões anuais e respectivos K_i
            emi_anuais = df['Emissão'].tolist()
            iema = pd.read_parquet(f'{dir}\\Dados Externos\\IEMA.parquet')['Emissões'].tolist()
            
            df['K_i'] = list(map(lambda x, y: round(x / y, 3), emi_anuais, iema))
            df.to_parquet(f'{dir}\\Emissões Sintéticas\\Anuais.parquet', index=False)
        
        else:
            os.makedirs('Usinas Problemáticas', exist_ok=True)
            shutil.move(f'Dados Tratados\\{usina}', 'Usinas Problemáticas')