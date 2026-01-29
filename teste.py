from iema import anos

# Transformar todos os itens da lista "anos" para o tipo int
anos = map(int, anos)

# Ajustar a lista "anos" para incluir um ano antes e dois anos depois dos limites - Range não capta o último valor.
# É necessário dois anos depois devido ao inventário do IEMA.
anos = list(range(min(anos) - 1, max(anos) + 3))

print(anos)