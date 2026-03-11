# Criação da rede neural

K.clear_session()   #Limpar redes geradas anteriormente
rede = tf.keras.Sequential()

rede.add(tf.keras.layers.Dense(64,activation='relu'))
rede.add(tf.keras.layers.Dense(64,activation='relu'))
rede.add(tf.keras.layers.Dense(64,activation='relu'))
rede.add(tf.keras.layers.Dense(1))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
rede.compile(optimizer=opt, loss='mean_squared_error')


# Treinamento e execução da rede neural

# Inicializa a medição do erro quadrático médio
mse = tf.keras.metrics.MeanSquaredError()

# While para treinar a rede N vezes até atingir um erro satisfatório
tic = time.time()
i=0
#while True:
i+=1
rede.fit(x_train,y_train,epochs=2000,verbose=0)
y_pred = rede.predict(x_train)

mse.update_state(y_train, y_pred)
mse_result = mse.result().numpy()
print(f'MSE Geração {i}: {mse_result}')
    # if mse_result < 20:
    #     tac = time.time()
    #     break
    # tempo_exec = tac - tic
    # print(f'\nTempo de Execução: {tempo_exec}')

#Salvar os pesos no diretório
# if usina == '':
#     dir = 'G:\\Meu Drive\\Documentos UFSCar\\Iniciação Científica (Giovanni Grizante)\\Python\\Código'
#     rede.save_weights(f'{dir}\\Pesos_Geral.weights.h5')
# else:
#     dir = f'G:\\Meu Drive\\Documentos UFSCar\\Iniciação Científica (Giovanni Grizante)\\Python\\Termelétricas (PANDAS)\\{usina}\\Pesos'
#     if os.path.exists(dir):
#         if os.path.isfile(f'{dir}\\{usina}.weights.h5') == False:
#             rede.save_weights(f'{dir}\\{usina}.weights.h5')
#     else:
#         os.makedirs(f'{dir}')
#         rede.save_weights(f'{dir}\\{usina}.weights.h5')