Considerações para utilização dos códigos:
- Para execução do projeto, é necessário obter os dados de geração em base horária disponíveis em ONS Dados Abertos, em arquivo .parquet. Devido ao processo de cálculo dos deltas (Não utilizado neste trabalho) é necessário realizar o download dos dados de geração do ano anterior ao primeiro relatório IEMA e dados de geração do ano posterior ao último relatório IEMA.

- Ademais, também é necessário gerar as tabelas de dados obtidos nas últimas páginas dos inventários do IEMA.

- Todos estes arquivos deverão ser armazenados em suas respectivas pastas.

Sequência de Execução dos Códigos:
1 - iema.py
2 - ons.py
3 - armezanar.py
4 - ampl.py
5 - sinteticas.py
6 - tratamento_rede.py
7 - rede_neural.py
8 - graficos.py
9 - graficos_aux.py

OBS.: Os arquivos para execução dos cálculos de minização utilizam os deltas para cálculo dos valores dos coeficientes dinâmicos. Neste trabalho, foram mantidos tais cálculos, porém não serão utilizados. Caso desejar, comentar/excluir os cálculos destes coeficientes dinâmicos para uma maior velocidade de execução do algoritmo.
