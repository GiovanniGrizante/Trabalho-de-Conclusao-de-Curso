import iema, ons, armazenar, ampl, sinteticas
import tratamento_rede, rede_neural
import graficos, graficos_aux

# Ordem de execução dos arquivos.

# Etapa 1 - Tratamento e armazenamento dos dados IEMA e ONS para minimização.
iema.main()
ons.main()
armazenar.main()

#Etapa 2 - Execução da minimização e geração das emissões sintéticas.
ampl.main()
sinteticas.main()

# Etapa 3 - Tratamento dos dados, criação e execução da rede neural.
tratamento_rede.main()
rede_neural.main()

# Etapa 4 - Geração dos gráficos. 
graficos.main()
graficos_aux.main()