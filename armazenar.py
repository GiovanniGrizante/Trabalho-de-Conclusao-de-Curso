import os
from ons import ons 
from iema import anos, horas, usinas_iema

# Perguntar ao usuário o tipo de arquivo desejado
tipo_arquivo = input("Qual o tipo de arquivo que deseja salvar? (csv / parquet): ")

# Função para armazenar os dados no formato escolhido pelo usuário
def armazenar(dir, arq, arm):
    if os.path.exists(dir):
        if not os.path.isfile(f'{dir}\\{arq}'):
            if tipo_arquivo == 'csv':
                arm.to_csv(f'{dir}\\{arq}', index=False)
            elif tipo_arquivo == 'parquet':
                arm.to_parquet(f'{dir}\\{arq}', index=False)
    else:
        os.makedirs(f'{dir}')
        if tipo_arquivo == 'csv':
            arm.to_csv(f'{dir}\\{arq}', index=False)
        elif tipo_arquivo == 'parquet':
            arm.to_parquet(f'{dir}\\{arq}', index=False)

usinas_faltando = []  # Inicializar o vetor fora do loop principal

for usina in usinas_iema:
    start_idx = 0
    dir = f'G:\\Meu Drive\\Documentos UFSCar\\TCC\\Dados Tratados\\{usina}\\Dados de Geração'

    for j, ano in enumerate(anos):

        arq = f'{ano}.{tipo_arquivo}'  # Ajustar a extensão do arquivo com base na escolha do usuário
        
        # Índice final para o ano atual
        end_idx = start_idx + horas[j]
        
        # Armazene os dados do ano no dicionário
        try:
          arm = ons[usina].iloc[start_idx:end_idx]
        except KeyError:
          usinas_faltando.append(usina)
          continue
        
        # Atualize o índice inicial para o próximo ano
        start_idx = end_idx
        armazenar(dir, arq, arm)

usinas_faltando = list(set(usinas_faltando))  # Remover duplicatas da lista

if usinas_faltando:
    print(f'Foram identificadas {len(usinas_faltando)} usinas faltando. Verificar o arquivo "Usinas Faltando.txt" para mais detalhes.')
    
    output_dir = 'G:\\Meu Drive\\Documentos UFSCar\\TCC\\Dados Tratados'  # Diretório específico para salvar o arquivo
    os.makedirs(output_dir, exist_ok=True)  # Criar o diretório, se não existir
    output_file = os.path.join(output_dir, 'Usinas Faltando.txt')

    with open(output_file, 'w') as f:
        for usina in usinas_faltando:
            f.write(f'{usina}\n')
