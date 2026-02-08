import pandas as pd
import os

# Este código acessa os dados da IEMA armazenados em planilhas Excel, organiza as informações e identifica as usinas térmicas que possuem dados de emissão para todos os anos estudados.

metodo = 'CEG'  # Definir o método de identificação das usinas (pode ser 'CEG' ou 'Usina')
relacao = {'CEG':'ceg', 'Usina':'nom_usina'}

dir = 'IEMA/Tabelas'

anos = sorted([os.path.splitext(ano)[0] for ano in set(os.listdir(dir)) - {'desktop.ini'}])   #Listar e ordenar os arquivos de anos na pasta IEMA
horas = {}  # Vetor para definir horas (linhas) dos anos estudados

# Criar um dicionário para armazenar as usinas que emitiram poluentes e fazer a correspondência entre os anos
iema = {}
usinas = []  # Lista para armazenar os conjuntos de usinas de cada ano

for ano in anos:
  horas[ano] = 8784 if int(ano) % 4 == 0 else 8760
  iema[ano] = pd.read_excel(f'{dir}\\{ano}.xlsx', sheet_name=0)
  usinas.append(set(iema[ano][metodo].unique()))  # Adiciona o conjunto de usinas do ano atual à lista - Comando SET organiza em conjuntos na lista
usinas_iema = set.intersection(*usinas)  # Calcula a interseção das usinas que aparecem em todos os anos

emissoes = {usina: [] for usina in usinas_iema}  # Dicionário para armazenar os valores de emissão anual de cada usina
for ano in anos:
  for usina in usinas_iema:
    emissoes[usina].append(iema[ano].loc[iema[ano][metodo] == usina, 'Emissões de Gases [tCO2]'].values[0])
