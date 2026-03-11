import pandas as pd
import os
import numpy as np
from iema import anos, horas

for usina in os.listdir('Dados Tratados'):
    horas_expandidas = []
    anos_expandidos = []

    PG = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Dados Externos', 'ONS.parquet'))['Geração']
    emissao = []

    if os.path.isfile(os.path.join('Dados Tratados', usina, 'Emissões Sintéticas', 'Coeficientes.parquet')):
        tab = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Emissões Sintéticas', 'Coeficientes.parquet'))

        alpha = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Alpha'].item()
        beta = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Beta'].item()
        gamma = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Gamma'].item()
        omega = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Omega'].item()
        mu = tab['Valores Estáticos'].loc[tab['Coeficientes'] == 'Mu'].item()

    else:
        raise FileNotFoundError(
            f'Arquivo de coeficientes para a usina {usina} não encontrado.')

    # Cálculo dos valores de emissão sintético conforme equação do artigo
    for k in range(len(PG)):
        emissao.append(round((alpha + beta * (PG[k]/100) + gamma * (PG[k]/100)**2) + omega * np.exp(mu * (PG[k]/100)), 3))
        
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
    
    dir = os.path.join('Dados Tratados', usina, 'Emissões Sintéticas')
    df.to_parquet(f'{dir}\\Horárias.parquet', index=False)
    df.groupby('Ano')['Emissão'].sum().reset_index().to_parquet(f'{dir}\\Anuais.parquet', index=False)