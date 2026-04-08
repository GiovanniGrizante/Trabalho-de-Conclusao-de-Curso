import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configurar dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Função para realizar o janelamento dos dados com DOIS LOOKBACKS
def janelamento_duplo(frame: pd.DataFrame, dyn_cols, static_cols, target_col,
                       lookback_longo, lookback_curto, horizon):

    # Transforma as colunas em matrizes numpy para facilitar o janelamento
    dyn = frame[dyn_cols].to_numpy(dtype="float32")      # [N, n_dyn]
    sta = frame[static_cols].to_numpy(dtype="float32")   # [N, n_static]
    tgt = frame[target_col].to_numpy(dtype="float32")    # [N]

    # Janelamento: para cada tempo t, pegamos os lookbacks das dinâmicas e o vetor estático no tempo t
    X_seq_longo, X_seq_curto, X_sta, y = [], [], [], []
    N = len(frame)
    
    for t in range(lookback_longo, N - horizon + 1):
        # Sequência LONGA (lookback_longo horas)
        X_seq_longo.append(dyn[t-lookback_longo:t, :])
        # Sequência CURTA (lookback_curto horas) - últimas horas da sequência longa
        X_seq_curto.append(dyn[t-lookback_curto:t, :])
        X_sta.append(sta[t, :])
        y.append(tgt[t:t+horizon])

    return (np.stack(X_seq_longo, axis=0),
            np.stack(X_seq_curto, axis=0),
            np.stack(X_sta, axis=0),
            np.stack(y, axis=0))

# Definição do modelo com DUAS GRUs (lookback longo e curto)
def criar_modelo_duplo(n_dyn, n_sta, horizon):
    """
    Cria o modelo da rede neural com duas GRUs paralelas:
    - Uma para lookback LONGO (tendências suaves)
    - Outra para lookback CURTO (transições abruptas)
    """
    
    class ModeloGRUDuplo(nn.Module):
        def __init__(self):
            super(ModeloGRUDuplo, self).__init__()
            
            # ===== GRU 1: LOOKBACK LONGO (suave) =====
            self.gru_longo1 = nn.GRU(n_dyn, 128, batch_first=True, bidirectional=True)
            self.dropout_longo1 = nn.Dropout(0.20)
            self.gru_longo2 = nn.GRU(256, 64, batch_first=True, bidirectional=True)
            self.dropout_longo2 = nn.Dropout(0.20)
            
            # ===== GRU 2: LOOKBACK CURTO (abrupto) =====
            self.gru_curto1 = nn.GRU(n_dyn, 64, batch_first=True, bidirectional=True)
            self.dropout_curto1 = nn.Dropout(0.20)
            self.gru_curto2 = nn.GRU(128, 32, batch_first=True, bidirectional=True)
            self.dropout_curto2 = nn.Dropout(0.20)
            
            # ===== Camadas para dados estáticos =====
            self.batch_norm = nn.BatchNorm1d(n_sta)
            self.dense_static = nn.Linear(n_sta, 32)
            
            # ===== Camadas combinadas =====
            # Saída da GRU longa: 64 (bidirecional -> 128, mas pegamos último)
            # Saída da GRU curta: 32 (bidirecional -> 64, mas pegamos último)
            # Saída estática: 32
            # Total: 128 + 64 + 32 = 224
            self.dense1 = nn.Linear(128 + 64 + 32, 64)
            self.dropout_combined = nn.Dropout(0.20)
            self.dense2 = nn.Linear(64, 64)
            self.output = nn.Linear(64, horizon)
            
            self.relu = nn.ReLU()
            
        def forward(self, x_seq_longo, x_seq_curto, x_sta):
            # ===== CAMINHO LONGO (suave) =====
            out_l, _ = self.gru_longo1(x_seq_longo)
            out_l = self.dropout_longo1(out_l)
            out_l, _ = self.gru_longo2(out_l)
            out_l = out_l[:, -1, :]  # [batch, 128]
            
            # ===== CAMINHO CURTO (abrupto) =====
            out_c, _ = self.gru_curto1(x_seq_curto)
            out_c = self.dropout_curto1(out_c)
            out_c, _ = self.gru_curto2(out_c)
            out_c = out_c[:, -1, :]  # [batch, 64]
            
            # ===== DADOS ESTÁTICOS =====
            s = self.batch_norm(x_sta)
            s = self.relu(self.dense_static(s))  # [batch, 32]
            
            # ===== COMBINAÇÃO (rede aprende os pesos automaticamente) =====
            combined = torch.cat([out_l, out_c, s], dim=1)  # [batch, 128+64+32=224]
            combined = self.relu(self.dense1(combined))
            combined = self.dropout_combined(combined)
            combined = self.relu(self.dense2(combined))
            output = self.output(combined)
            
            return output
    
    model = ModeloGRUDuplo()
    return model

# Função de treinamento (adaptada para o modelo duplo)

def treinar_modelo_duplo(
    model,
    train_loader,
    val_loader,
    previsao,
    dyn_cols,                     # Para localizar PG
    pg_mean,                      # Para desnormalizar PG
    pg_std,                       # Para desnormalizar PG
    epochs=100,
    patience=10,
    huber_delta=0.5,
    peso_transicao=8.0,
    clip_grad=0.5,
    lambda_fis=0.2                # Peso da física
):
    """
    Treina o modelo com Early Stopping e ReduceLR.
    Caso previsao == '2', adiciona termo Physics-Informed na loss.
    """
    model = model.to(device)

    # Transforma os dados de normalização em tensores
    pg_mean = torch.tensor(pg_mean, device=device)
    pg_std  = torch.tensor(pg_std, device=device)

    # =============================
    # PARÂMETROS FÍSICOS TREINÁVEIS
    # =============================
    alpha = nn.Parameter(torch.tensor(0.1, device=device))
    beta  = nn.Parameter(torch.tensor(0.1, device=device))
    gamma = nn.Parameter(torch.tensor(0.01, device=device))
    omega = nn.Parameter(torch.tensor(0.05, device=device))
    mu    = nn.Parameter(torch.tensor(1e-4, device=device))     # Iniciar com valor pequeno para evitar explosão do resultado

    # =============================
    # Função física (equação guia)
    # =============================
    def emissao_fisica(PG):
        """
        PG: tensor [batch] ou [batch, horizon]
        Retorna emissões físicas >= 0
        """
        emissao = (
            alpha
            + beta * PG
            + gamma * PG**2
            + omega * torch.exp(mu * PG)
        )
        return torch.relu(emissao)  # não-negatividade física

    # =============================
    # OPTIMIZER (modelo + física)
    # =============================
    optimizer = optim.Adam(
        list(model.parameters()) + [alpha, beta, gamma, omega, mu],
        lr=1e-3
    )

    criterion_huber = nn.HuberLoss(delta=huber_delta)
    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {
        'loss': [], 'mae': [], 'mse': [],
        'val_loss': [], 'val_mae': [], 'val_mse': []
    }

    idx_pg = dyn_cols.index("Geração")  # índice do PG nas dinâmicas

    for epoch in range(epochs):

        # =============================
        # TREINAMENTO
        # =============================
        model.train()
        train_loss = train_mae = train_mse = 0.0

        for x_seq_longo, x_seq_curto, x_sta, y in train_loader:
            x_seq_longo = x_seq_longo.to(device)
            x_seq_curto = x_seq_curto.to(device)
            x_sta = x_sta.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x_seq_longo, x_seq_curto, x_sta)

            # ===== LOSS DE DADOS =====
            loss_data = criterion_huber(output, y)

            # ===== PESOS MAIORES EM TRANSIÇÕES =====
            is_transicao = x_sta[:, 0] == 1
            weight = torch.where(
                is_transicao,
                torch.tensor(peso_transicao, device=device),
                torch.tensor(1.0, device=device)
            )
            loss_data = (loss_data * weight).mean()

            # ===== LOSS FÍSICA (PINN) =====
            if previsao == '2':
                # PG no instante de previsão
                PG = x_seq_curto[:, -1, idx_pg]  # [batch]

                # Ajuste para horizonte > 1
                if output.ndim == 2 and output.shape[1] > 1:
                    PG = PG.unsqueeze(1).repeat(1, output.shape[1])

                # Desnormalização dos dados de geração
                PG = PG * pg_std + pg_mean

                emissao_fis = emissao_fisica(PG)

                loss_fis = nn.MSELoss()(output, emissao_fis)

                loss = loss_data + lambda_fis * loss_fis
            else:
                loss = loss_data

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            train_loss += loss.item()
            train_mae += criterion_mae(output, y).item()
            train_mse += criterion_mse(output, y).item()

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_mse /= len(train_loader)

        # =============================
        # VALIDAÇÃO
        # =============================
        model.eval()
        val_loss = val_mae = val_mse = 0.0

        with torch.no_grad():
            for x_seq_longo, x_seq_curto, x_sta, y in val_loader:
                x_seq_longo = x_seq_longo.to(device)
                x_seq_curto = x_seq_curto.to(device)
                x_sta = x_sta.to(device)
                y = y.to(device)

                output = model(x_seq_longo, x_seq_curto, x_sta)

                loss_val = criterion_huber(output, y).mean()

                val_loss += loss_val.item()
                val_mae += criterion_mae(output, y).item()
                val_mse += criterion_mse(output, y).item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)

        history['loss'].append(round(train_loss, 3))
        history['mae'].append(round(train_mae, 3))
        history['mse'].append(round(train_mse, 3))
        history['val_loss'].append(round(val_loss, 3))
        history['val_mae'].append(round(val_mae, 3))
        history['val_mse'].append(round(val_mse, 3))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# Função para avaliar o modelo no teste (adaptada)
def avaliar_modelo_duplo(model, test_loader):
    """
    Avalia o modelo no conjunto de teste e retorna as predições
    """
    model.eval()
    test_loss = 0
    test_mae = 0
    test_mse = 0
    
    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for x_seq_longo, x_seq_curto, x_sta, y in test_loader:
            x_seq_longo = x_seq_longo.to(device)
            x_seq_curto = x_seq_curto.to(device)
            x_sta = x_sta.to(device)
            y = y.to(device)
            
            output = model(x_seq_longo, x_seq_curto, x_sta)
            
            test_loss += criterion_mae(output, y).item()
            test_mae += criterion_mae(output, y).item()
            test_mse += criterion_mse(output, y).item()
            
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(output.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    test_mse /= len(test_loader)
    
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)
    
    return test_loss, test_mae, test_mse, y_true_all, y_pred_all


def obter_iniciais(mensagem, opcoes_validas):
    while True:
        try:
            valor = int(input(mensagem))
            if valor in opcoes_validas:
                return valor
            print(f"Por favor, digite um número válido.")
        except ValueError:
            print("Por favor, digite um número válido.")


def main(previsao):
    # Pergunta ao usuário se deseja retreinar os modelos ou usar os já treinados.
    treinar = bool(obter_iniciais('Retreinar modelos? (1 - Sim | 0 - Não): ', [0, 1]))
    # Pergunta ao usuário quais os valores de janela de previsão.
    horizon = obter_iniciais('Qual a janela de previsão? (1h ou 24h): ', [1, 24])
    
    # Parâmetros dos lookbacks
    LOOKBACK_LONGO = 336  # 14 dias (tendências suaves)
    LOOKBACK_CURTO = 48   # 3 dias (transições abruptas)
    
    # Loop para cada usina
    for usina in os.listdir('Usinas'):
        
        # Carregar dados
        if previsao == '1':
            dir = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Dados')
        else:
            dir = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Dados')

        df_tr = pd.read_parquet(os.path.join(dir, 'Treino.parquet'))
        df_val = pd.read_parquet(os.path.join(dir, 'Validação.parquet'))
        df_te = pd.read_parquet(os.path.join(dir, 'Teste.parquet'))
        df_scaler = pd.read_parquet(os.path.join(dir, 'Scaler Geração.parquet'))

        # Carregar dados para desnormalização e transformá-los em tensores
        pg_mean = float(df_scaler.loc[0, 'Mean'])
        pg_std = float(df_scaler.loc[0, 'Std'])

        # Coluna alvo
        target_col = 'Emissão'

        # Dinâmicas
        dyn_cols = (
            [c for c in df_tr.columns if c.startswith("Categoria de geração_")] +
            ["Geração", "Duração da Transição", "Fase da Transição",
             "Seno Índice", "Cosseno Índice"]
        )

        # Estáticas
        static_cols = (
            ["Potência Instalada", "Eficiência Energética [%]", "Fator de Capacidade [%]"] +
            [c for c in df_tr.columns if c.startswith("Combustível_")] +
            [c for c in df_tr.columns if c.startswith("Ciclo de Operação_")]
        )

        # Aplicar janelamento DUPLO
        Xtr_longo, Xtr_curto, Xtr_sta, ytr = janelamento_duplo(
            df_tr, dyn_cols, static_cols, target_col,
            lookback_longo=LOOKBACK_LONGO, lookback_curto=LOOKBACK_CURTO, horizon=horizon
        )
        Xva_longo, Xva_curto, Xva_sta, yva = janelamento_duplo(
            df_val, dyn_cols, static_cols, target_col,
            lookback_longo=LOOKBACK_LONGO, lookback_curto=LOOKBACK_CURTO, horizon=horizon
        )
        Xte_longo, Xte_curto, Xte_sta, yte = janelamento_duplo(
            df_te, dyn_cols, static_cols, target_col,
            lookback_longo=LOOKBACK_LONGO, lookback_curto=LOOKBACK_CURTO, horizon=horizon
        )

        # Obter dimensões
        n_dyn = Xtr_longo.shape[2]
        n_sta = Xtr_sta.shape[1]
        horizon_real = ytr.shape[1]

        # Criar diretórios para salvar resultados
        if previsao == '1':
            modelo_dir = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Modelos')
            historico_dir = os.path.join('Usinas', usina, 'Minimização', 'Rede Neural', 'Históricos')
        else:
            modelo_dir = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Modelos')
            historico_dir = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Históricos')

        os.makedirs(modelo_dir, exist_ok=True)
        os.makedirs(historico_dir, exist_ok=True)

        modelo_path = os.path.join(modelo_dir, "best_model.pth")
        history_path = os.path.join(historico_dir, "Treino.parquet")
        teste_path = os.path.join(historico_dir, "Teste.parquet")

        # Converter dados para tensores PyTorch (agora com 4 componentes)
        train_dataset = TensorDataset(
            torch.FloatTensor(Xtr_longo),
            torch.FloatTensor(Xtr_curto),
            torch.FloatTensor(Xtr_sta),
            torch.FloatTensor(ytr)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(Xva_longo),
            torch.FloatTensor(Xva_curto),
            torch.FloatTensor(Xva_sta),
            torch.FloatTensor(yva)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(Xte_longo),
            torch.FloatTensor(Xte_curto),
            torch.FloatTensor(Xte_sta),
            torch.FloatTensor(yte)
        )

        # Criar dataloaders
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        if Xtr_longo.shape[0] == 0:
            continue  # Pula esta usina

        if os.path.exists(modelo_path) and not treinar:
            # Carregar modelo salvo
            # print(f"  Carregando modelo existente...")
            model = criar_modelo_duplo(n_dyn, n_sta, horizon_real)
            model.load_state_dict(torch.load(modelo_path, map_location=device))
            model = model.to(device)
        else:
            # Criar e treinar novo modelo
            model = criar_modelo_duplo(n_dyn, n_sta, horizon_real)
            
            model, history = treinar_modelo_duplo(
                model,
                train_loader,
                val_loader,
                previsao=previsao,
                dyn_cols=dyn_cols,
                pg_mean = pg_mean,
                pg_std = pg_std
            )

            # Salvar modelo
            torch.save(model.state_dict(), modelo_path)
            
            # Salvar histórico
            history_df = pd.DataFrame(history)
            history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
            history_df.to_parquet(history_path, index=False)

        # Avaliar no teste
        test_loss, test_mae, test_mse, y_true_all, y_pred_all = avaliar_modelo_duplo(model, test_loader)
        
        if horizon == 1:
            y_true_flat = y_true_all.flatten().astype(float)
            y_pred_flat = y_pred_all.flatten().astype(float)
            
            predicoes_df = pd.DataFrame({
                'Emissões Calculadas': np.round(y_true_flat, 3),
                'Emissões Previstas': np.round(y_pred_flat, 3)
            })
        else:
            y_true_list = [np.round(np.array(x).flatten(), 3) for x in y_true_all]
            y_pred_list = [np.round(np.array(x).flatten(), 3) for x in y_pred_all]
            
            predicoes_df = pd.DataFrame({
                'Emissões Calculadas': y_true_list,
                'Emissões Previstas': y_pred_list
            })
        
        predicoes_path = os.path.join(historico_dir, "Predições.parquet")
        predicoes_df.to_parquet(predicoes_path, index=False)
        
        # Salvar resultados do teste
        teste_df = pd.DataFrame([{
            'usina': usina,
            'loss': round(test_loss,3),
            'mae': round(test_mae,3),
            'mse': round(test_mse,3)
        }])
        teste_df.to_parquet(teste_path, index=False)