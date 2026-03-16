import os, pandas as pd, numpy as np, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Função para realizar o janelamento dos dados, transformando as séries temporais em matrizes de entrada para a rede neural
def janelamento(frame: pd.DataFrame, dyn_cols, static_cols, target_col,
                lookback=168, horizon=1):

    # Transforma as colunas em matrizes numpy para facilitar o janelamento
    dyn = frame[dyn_cols].to_numpy(dtype="float32")      # [N, n_dyn]
    sta = frame[static_cols].to_numpy(dtype="float32")   # [N, n_static]
    tgt = frame[target_col].to_numpy(dtype="float32")    # [N]

    # Janelamento: para cada tempo t, pegamos os lookback (passos anteriores) das dinâmicas e o vetor estático no tempo t
    # A variável alvo é o valor no tempo t+horizon (horizon=1 para prever a próxima hora)
    X_seq, X_sta, y = [], [], []
    N = len(frame)
    for t in range(lookback, N - horizon + 1):      # começa em lookback para garantir que temos dados anteriores, e termina em N-horizon para garantir que temos o alvo
        X_seq.append(dyn[t-lookback:t, :])          # pega 168 horas anteriores
        X_sta.append(sta[t, :])                     # vetor estático no tempo t
        y.append(tgt[t:t+horizon])                  # prevemos t+1 (horizon=1)

    return (np.stack(X_seq, axis=0),    # [N-lookback-horizon+1, lookback, n_dyn]
            np.stack(X_sta, axis=0),
            np.stack(y,     axis=0))

# Função para criar o modelo da rede neural, definir os callbacks e compilar o modelo
def modelo_rede_neural(usina, lookback, n_dyn, n_sta, horizon):
    # Arquitetura do modelo da rede neural (GRU + Dense)
    # Arquitetura GRU considera dados temporais.
    # GRU - Bidirecional - Não considera dados em tempo real.
    # GRU - Causal - Considera dados em tempo real, mas tem desempenho inferior.
    # Dense - Camada para inserção dos dados estáticos.

    # Entrada para as dinâmicas (sequência temporal)
    input_seq = keras.Input(shape=(lookback, n_dyn), name="input_seq")

    # Camada GRU bidirecional para capturar padrões temporais
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(input_seq) # return_sequences=True para manter a sequência para a próxima camada
    x = layers.Dropout(0.20)(x) # Dropout para evitar overfitting (20% de neurônios desligados durante o treinamento)
    x = layers.Bidirectional(layers.GRU(64))(x)  # embedding temporal final

    # Entrada para as variáveis estáticas
    input_stat = keras.Input(shape=(n_sta,), name="static") # shape=(n_sta,) porque é um vetor, não uma sequência
    s = layers.BatchNormalization()(input_stat)
    s = layers.Dense(32, activation="relu")(s)

    # Combina as saídas da GRU (dados temporais) e da Dense (dados estáticos)
    z = layers.Concatenate()([x, s])
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.20)(z) # Dropout para evitar overfitting (20% de neurônios desligados durante o treinamento)
    out = layers.Dense(horizon, name="emissao")(z)  # horizon=1 => saída escalar

    # Criação do modelo Keras com as duas entradas (sequência e estática) e a saída (previsão de emissão)
    model = keras.Model(inputs=[input_seq, input_stat], outputs=out)

    # Compilação do modelo com otimizador Adam e função de perda MAE (Erro Absoluto Médio)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",     # MAE - Erro Absoluto Médio ; MSE - Erro Quadrático Médio
        metrics=[keras.metrics.MAE, keras.metrics.MSE]
    )

    # Definir callbacks para salvar o melhor modelo e reduzir a taxa de aprendizado se a validação não melhorar
    # ReduceLROnPlateau: reduz a taxa de aprendizado se a métrica monitorada (val_loss) não melhorar por um número específico de épocas (patience).
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=0     # Exibe o progresso do treinamento a cada época (0 = sem saída, 1 = barra de progresso, 2 = uma linha por época)
    )

    # EarlyStopping: para o treinamento se a métrica monitorada (val_loss) não melhorar por um número específico de épocas (patience), 
    # e restaura os pesos do melhor modelo encontrado durante o treinamento.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0  # Exibe o progresso do treinamento a cada época (0 = sem saída, 1 = barra de progresso, 2 = uma linha por época)
    )

    # Criação do diretório para salvar os modelos, se não existir
    os.makedirs(os.path.join("Dados Tratados", usina, 'Rede Neural', 'Modelos'), exist_ok=True)

    # ModelCheckpoint: salva o modelo em um arquivo (modelo_best.h5) sempre que a métrica monitorada (val_loss) melhorar.
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("Dados Tratados", usina, 'Rede Neural', 'Modelos', "Best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=0     # Exibe o progresso do treinamento a cada época (0 = sem saída, 1 = barra de progresso, 2 = uma linha por época)
    )

    return model, reduce_lr, early_stop, ckpt


def main():
    # Pergunta ao usuário se deseja retreinar os modelos ou usar os já treinados
    while True:
        try:
            treinar = bool(int(input("Retreinar modelos? (1 - Sim | 0 - Não): ")))
            break
        except ValueError:
            print("Entrada inválida. Por favor, insira 1 para Sim ou 0 para Não.\n")

    # Loop para cada usina: carrega os dados, realiza o janelamento, treina o modelo (se necessário) e avalia no teste
    for usina in os.listdir('Dados Tratados'):
        df_tr = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados', 'Treino.parquet'))
        df_val = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados', 'Validação.parquet'))
        df_te = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados', 'Teste.parquet'))

        # Coluna alvo: o que queremos prever
        target_col = 'Emissão'

        # Dinâmicas: variáveis que mudam ao longo do tempo
        dyn_cols = (
            [c for c in df_tr.columns if c.startswith("Categoria de geração_")] +
            ["Geração", "Duração da Transição", "Fase da Transição",
            "Seno Índice", "Cosseno Índice"]
        )

        # Estáticas: constantes/one-hot nos dados originais
        static_cols = (
            ["Potência Instalada", "Eficiência Energética [%]", "Fator de Capacidade [%]"] +
            [c for c in df_tr.columns if c.startswith("Combustível_")] +
            [c for c in df_tr.columns if c.startswith("Ciclo de Operação_")]
        )

        Xtr_seq, Xtr_sta, ytr = janelamento(df_tr, dyn_cols, static_cols, target_col)
        Xva_seq, Xva_sta, yva = janelamento(df_val, dyn_cols, static_cols, target_col)
        Xte_seq, Xte_sta, yte = janelamento(df_te, dyn_cols, static_cols, target_col)

        lookback = Xtr_seq.shape[1]   # Período de observação (Alterar na função "janelamento")
        horizon = ytr.shape[1]        # Período de previsão (Alterar na função "janelamento")
        n_dyn = Xtr_seq.shape[2]      # Número de variáveis dinâmicas
        n_sta = Xtr_sta.shape[1]      # Número de variáveis estáticas

        os.makedirs(os.path.join("Dados Tratados", usina, 'Rede Neural', 'Históricos'), exist_ok=True)

        if os.path.exists(os.path.join("Dados Tratados", usina, 'Rede Neural', 'Modelos', "Best.keras")) and treinar == False:
            model = load_model(os.path.join("Dados Tratados", usina, 'Rede Neural', 'Modelos', "Best.keras"))
        else:
            model, reduce_lr, early_stop, ckpt = modelo_rede_neural(usina, lookback, n_dyn, n_sta, horizon)

            treino = model.fit(
                x=[Xtr_seq, Xtr_sta], y=ytr,
                validation_data=([Xva_seq, Xva_sta], yva),
                epochs=100,     # Número máximo de épocas para o treinamento (pode ser interrompido pelo EarlyStopping)
                batch_size=256, # Tamanho do lote para o treinamento (número de amostras processadas antes de atualizar os pesos do modelo)
                callbacks=[reduce_lr, early_stop, ckpt],
                verbose=0   # Exibe o progresso do treinamento a cada época (0 = sem saída, 1 = barra de progresso, 2 = uma linha por época)
            )

            treino_df = pd.DataFrame(treino.history)  # loss, mae, mse, val_loss, val_mae, val_mse...
            treino_df.insert(0, "epoch", np.arange(1, len(treino_df) + 1))  # Adiciona a coluna "epoch"
            treino_df.to_parquet(os.path.join("Dados Tratados", usina, 'Rede Neural', 'Históricos', "Treino.parquet"), index=False)

        # Avaliação final no teste
        teste = model.evaluate([Xte_seq, Xte_sta], yte, verbose=0)
        teste_df = pd.DataFrame([teste], columns=model.metrics_names)  # loss, mae, mse
        teste_df.insert(0, "usina", usina)  # Adiciona a coluna "usina"
        teste_df.to_parquet(os.path.join("Dados Tratados", usina, 'Rede Neural', 'Históricos', "Teste.parquet"), index=False)