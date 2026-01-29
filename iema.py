import pandas as pd
import os

# Este código acessa os dados da IEMA armazenados em planilhas Excel, organiza as informações e identifica as usinas térmicas que possuem dados de emissão para todos os anos estudados.

metodo = input("Método de separação das usinas térmicas (1 - CEG / 2 - Nome da Usina): ")

if metodo == '1':
    metodo = 'CEG'
elif metodo == '2':
    metodo = 'Usina'

dir = 'G:\\Meu Drive\\Documentos UFSCar\\TCC\\IEMA\\Tabelas'

anos = sorted([os.path.splitext(ano)[0] for ano in set(os.listdir(dir)) - {'desktop.ini'}])   #Listar e ordenar os arquivos de anos na pasta IEMA
horas = []  # Vetor para definir horas (linhas) dos anos estudados

for ano in anos:
  horas.append(8784) if int(ano) % 4 == 0 else horas.append(8760)

iema = {}
usinas_anos = []  # Lista para armazenar os conjuntos de usinas de cada ano

for ano in anos:
  iema[ano] = pd.read_excel(f'{dir}\\{ano}.xlsx', sheet_name=0)  # Armazena diretamente a segunda aba na variável
  usinas_anos.append(set(iema[ano][metodo].unique()))  # Adiciona o conjunto de usinas do ano atual à lista - Comando SET organiza em conjuntos na lista

usinas_iema = set.intersection(*usinas_anos)  # Calcula a interseção das usinas que aparecem em todos os anos