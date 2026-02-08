import os, pandas as pd
from ons import ons
from iema import emissoes, anos, horas

# Perguntar ao usuário o tipo de arquivo desejado
tipo_arquivo = input("Qual o tipo de arquivo que deseja salvar? (1 - parquet / 2 - dat / 3 - Ambos): ")

# Função para armazenar os dados no formato escolhido pelo usuário
def criar_parquet(dir_base, usina, tab_ons, tab_iema):
    dir = os.path.join(dir_base, 'Dados Tratados', usina, 'Dados de Geração')

    if os.path.exists(dir):
        if not os.path.isfile(f'{dir}\\ONS.parquet'):
            tab_ons.to_parquet(f'{dir}\\ONS.parquet', index=False)
        if not os.path.isfile(f'{dir}\\IEMA.parquet'):
            tab_iema.to_parquet(f'{dir}\\IEMA.parquet', index=False)
    else:
        os.makedirs(f'{dir}')
        tab_ons.to_parquet(f'{dir}\\ONS.parquet', index=False)
        tab_iema.to_parquet(f'{dir}\\IEMA.parquet', index=False)

def criar_dat(dir_base, usina, horas, tab_ons, tab_iema):
    dir = os.path.join(dir_base, 'AMPL', 'Usinas', usina)

    dat_content = 'data;\n\n'

    # Anos
    dat_content += f"set ANO := {tab_iema['Ano'].str.cat(sep=' ')};\n\n"

    # Horas respectivas a cada ano
    dat_content += f"param H :=\n"
    for ano in tab_iema['Ano']:
        dat_content += f'{ano} {horas[ano]}\n'
    dat_content += ';\n\n'

    # Formatar os dados para o arquivo .dat
    # Emissões
    dat_content += f"param  emissoes :=\n"
    for _, row in tab_iema.iterrows():
        dat_content += f"{row['Ano']}  {row['Emissões']}\n"
    dat_content += ";\n\n"

    # Geração e Deltas
    dat_content += f"param:  pg   delta_pg_ant   delta_pg_pos :=\n"
    for _, row in tab_ons.iterrows():
        dat_content += f"[{row['Ano']},{row['Índice']}]  {row['Geração']}  {row['Delta menos']}  {row['Delta mais']}\n"
    dat_content += ";"

    # Salvar arquivo .dat
    arq = os.path.join(dir, f'{usina}.dat')
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(arq, 'w') as f:
        f.write(dat_content)


dir_base = f'G:\\Meu Drive\\Documentos UFSCar\\TCC'

for usina in ons.keys():
    tab_ons = ons[usina]
    tab_iema = pd.DataFrame({'Ano': anos, 'Emissões': emissoes[usina]})

    if tipo_arquivo == '1':
        criar_parquet(dir_base, usina, tab_ons, tab_iema)
    elif tipo_arquivo == '2':
        criar_dat(dir_base, usina, horas, tab_ons, tab_iema)
    else:
        criar_parquet(dir_base, usina, tab_ons, tab_iema)
        criar_dat(dir_base, usina, horas, tab_ons, tab_iema)