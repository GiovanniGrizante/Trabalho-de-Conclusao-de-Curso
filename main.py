import iema, ons, armazenar, ampl, sinteticas
import tratamento_rede, rede_neural
import graficos, graficos_aux

# Ordem de execução dos arquivos.

# Etapa 1 - Tratamento e armazenamento dos dados IEMA e ONS para minimização.
etapa1 = input('Deseja executar a etapa 1? (1 - Sim | 2 - Não) ')
if etapa1 == '1':
    iema.main()
    ons.main()
    armazenar.main()
else:
    pass

#Etapa 2 - Execução da minimização e geração das emissões sintéticas.
etapa2 = input('Deseja executar a etapa 2? (1 - Sim | 2 - Não) ')
if etapa2 == '1':
    ampl.main()
    sinteticas.main()
else:
    pass

# Etapa 3 - Tratamento dos dados, criação e execução da rede neural.
etapa3 = input('Deseja executar a etapa 3? (1 - Sim | 2 - Não) ')
if etapa3 == '1':
    tratamento_rede.main()
    rede_neural.main()
else:
    pass

# Etapa 4 - Geração dos gráficos. 
etapa4 = input('Deseja executar a etapa 4? (1 - Sim | 2 - Não) ')
if etapa4 == '1':
    graficos.main()
    graficos_aux.main()
else:
    pass