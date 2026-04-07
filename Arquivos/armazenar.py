import os, pandas as pd

# Função para armazenar os dados no formato escolhido pelo usuário
def criar_parquet(usina, tab_ons, tab_iema):
    dir = os.path.join('Usinas', usina, 'Dados Externos')

    if os.path.exists(dir):
        tab_ons.to_parquet(f'{dir}\\ONS.parquet', index=False)
        tab_iema.to_parquet(f'{dir}\\IEMA.parquet', index=False)
    else:
        os.makedirs(f'{dir}')
        tab_ons.to_parquet(f'{dir}\\ONS.parquet', index=False)
        tab_iema.to_parquet(f'{dir}\\IEMA.parquet', index=False)

def criar_dat(usina, horas, tab_ons, tab_iema):
    dir = os.path.join('AMPL', 'Usinas', usina)

    dat_content = 'data;\n\n'

    # Anos
    dat_content += f"set ANO := {tab_iema['Ano'].str.cat(sep=' ')};\n\n"

    # Horas respectivas a cada ano
    dat_content += f"param H :=\n"
    for ano in tab_iema['Ano']:
        dat_content += f'{ano} {horas[ano]-1}\n'
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

def main(dados_iema, dados_ons):
    # Perguntar ao usuário o tipo de arquivo desejado
    tipo_arquivo = input("Qual o tipo de arquivo que deseja salvar? (1 - parquet / 2 - dat / 3 - Ambos): ")
    
    horas = dados_iema['horas']
    for usina in dados_ons.keys():
        tab_ons = dados_ons[usina]
        tab_iema = pd.DataFrame({'Ano': dados_iema['anos'], 'Emissões': dados_iema['emissoes'][usina]})

        if tipo_arquivo == '1':
            criar_parquet(usina, tab_ons, tab_iema)
        elif tipo_arquivo == '2':
            criar_dat(usina, horas, tab_ons, tab_iema)
        else:
            criar_parquet(usina, tab_ons, tab_iema)
            criar_dat(usina, horas, tab_ons, tab_iema)