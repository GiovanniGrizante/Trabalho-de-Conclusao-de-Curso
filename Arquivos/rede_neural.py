import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configurar dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Função para realizar o janelamento dos dados (mantida igual)
def janelamento(frame: pd.DataFrame, dyn_cols, static_cols, target_col,
                lookback=168, horizon=1):

    # Transforma as colunas em matrizes numpy para facilitar o janelamento
    dyn = frame[dyn_cols].to_numpy(dtype="float32")      # [N, n_dyn]
    sta = frame[static_cols].to_numpy(dtype="float32")   # [N, n_static]
    tgt = frame[target_col].to_numpy(dtype="float32")    # [N]

    # Janelamento: para cada tempo t, pegamos os lookback (passos anteriores) das dinâmicas e o vetor estático no tempo t
    X_seq, X_sta, y = [], [], []
    N = len(frame)
    for t in range(lookback, N - horizon + 1):
        X_seq.append(dyn[t-lookback:t, :])
        X_sta.append(sta[t, :])
        y.append(tgt[t:t+horizon])

    return (np.stack(X_seq, axis=0),
            np.stack(X_sta, axis=0),
            np.stack(y, axis=0))

# Definição do modelo (arquitetura da rede)
def criar_modelo(n_dyn, n_sta, horizon=1):
    """
    Cria o modelo da rede neural com a mesma arquitetura do TensorFlow
    """
    
    class ModeloGRU(nn.Module):
        def __init__(self):
            super(ModeloGRU, self).__init__()
            
            # Camadas para dados temporais
            self.gru1 = nn.GRU(n_dyn, 128, batch_first=True, bidirectional=True)
            self.dropout1 = nn.Dropout(0.20)
            self.gru2 = nn.GRU(256, 64, batch_first=True, bidirectional=True)
            
            # Camadas para dados estáticos
            self.batch_norm = nn.BatchNorm1d(n_sta)
            self.dense_static = nn.Linear(n_sta, 32)
            
            # Camadas combinadas
            self.dense1 = nn.Linear(128 + 32, 64)  # 128 da GRU (64*2) + 32 estáticas
            self.dropout2 = nn.Dropout(0.20)
            self.dense2 = nn.Linear(64, 64)
            self.output = nn.Linear(64, horizon)
            
            self.relu = nn.ReLU()
            
        def forward(self, x_seq, x_sta):
            # Processar sequência temporal
            out, _ = self.gru1(x_seq)
            out = self.dropout1(out)
            out, _ = self.gru2(out)
            
            # Pegar última saída da GRU
            out = out[:, -1, :]  # [batch, 128]
            
            # Processar dados estáticos
            s = self.batch_norm(x_sta)
            s = self.relu(self.dense_static(s))
            
            # Combinar
            combined = torch.cat([out, s], dim=1)
            combined = self.relu(self.dense1(combined))
            combined = self.dropout2(combined)
            combined = self.relu(self.dense2(combined))
            output = self.output(combined)
            
            return output
    
    model = ModeloGRU()
    return model

# Função de treinamento (equivalente ao model.fit)
def treinar_modelo(model, train_loader, val_loader, epochs=100, patience=10):
    """
    Treina o modelo com early stopping e reduce learning rate
    """
    model = model.to(device)
    
    # Otimizador
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Loss functions
    criterion = nn.L1Loss()  # MAE
    mse_loss = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )
    
    # Variáveis para early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Histórico de treinamento
    history = {
        'loss': [], 'mae': [], 'mse': [],
        'val_loss': [], 'val_mae': [], 'val_mse': []
    }
    
    for epoch in range(epochs):
        # === TREINAMENTO ===
        model.train()
        train_loss = 0
        train_mae = 0
        train_mse = 0
        
        for x_seq, x_sta, y in train_loader:
            x_seq = x_seq.to(device)
            x_sta = x_sta.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x_seq, x_sta)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += loss.item()
            train_mse += mse_loss(output, y).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_mse /= len(train_loader)
        
        # === VALIDAÇÃO ===
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        
        with torch.no_grad():
            for x_seq, x_sta, y in val_loader:
                x_seq = x_seq.to(device)
                x_sta = x_sta.to(device)
                y = y.to(device)
                
                output = model(x_seq, x_sta)
                loss = criterion(output, y)
                
                val_loss += loss.item()
                val_mae += loss.item()
                val_mse += mse_loss(output, y).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Salvar histórico
        history['loss'].append(train_loss)
        history['mae'].append(train_mae)
        history['mse'].append(train_mse)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restaurar melhores pesos
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# Função para avaliar o modelo no teste
def avaliar_modelo(model, test_loader):
    """
    Avalia o modelo no conjunto de teste
    """
    model.eval()
    test_loss = 0
    test_mae = 0
    test_mse = 0
    
    criterion = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    with torch.no_grad():
        for x_seq, x_sta, y in test_loader:
            x_seq = x_seq.to(device)
            x_sta = x_sta.to(device)
            y = y.to(device)
            
            output = model(x_seq, x_sta)
            test_loss += criterion(output, y).item()
            test_mae += criterion(output, y).item()
            test_mse += mse_loss(output, y).item()
    
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    test_mse /= len(test_loader)
    
    return test_loss, test_mae, test_mse


def main():
    # Pergunta ao usuário se deseja retreinar os modelos ou usar os já treinados
    while True:
        treinar = int(input("Retreinar modelos? (1 - Sim | 0 - Não): "))
        if treinar in [0, 1]:
            treinar = bool(treinar)
            break

    # Loop para cada usina
    for usina in os.listdir('Dados Tratados'):
        # Carregar dados
        df_tr = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados', 'Treino.parquet'))
        df_val = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados', 'Validação.parquet'))
        df_te = pd.read_parquet(os.path.join('Dados Tratados', usina, 'Rede Neural', 'Dados', 'Teste.parquet'))

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

        # Aplicar janelamento
        Xtr_seq, Xtr_sta, ytr = janelamento(df_tr, dyn_cols, static_cols, target_col)
        Xva_seq, Xva_sta, yva = janelamento(df_val, dyn_cols, static_cols, target_col)
        Xte_seq, Xte_sta, yte = janelamento(df_te, dyn_cols, static_cols, target_col)

        # Obter dimensões
        n_dyn = Xtr_seq.shape[2]
        n_sta = Xtr_sta.shape[1]
        horizon = ytr.shape[1]

        # Criar diretórios para salvar resultados
        modelo_dir = os.path.join("Dados Tratados", usina, 'Rede Neural', 'Modelos')
        historico_dir = os.path.join("Dados Tratados", usina, 'Rede Neural', 'Históricos')
        os.makedirs(modelo_dir, exist_ok=True)
        os.makedirs(historico_dir, exist_ok=True)

        modelo_path = os.path.join(modelo_dir, "best_model.pth")
        history_path = os.path.join(historico_dir, "Treino.parquet")
        teste_path = os.path.join(historico_dir, "Teste.parquet")

        # Converter dados para tensores PyTorch
        train_dataset = TensorDataset(
            torch.FloatTensor(Xtr_seq),
            torch.FloatTensor(Xtr_sta),
            torch.FloatTensor(ytr)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(Xva_seq),
            torch.FloatTensor(Xva_sta),
            torch.FloatTensor(yva)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(Xte_seq),
            torch.FloatTensor(Xte_sta),
            torch.FloatTensor(yte)
        )

        # Criar dataloaders
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        if os.path.exists(modelo_path) and not treinar:
            # Carregar modelo salvo
            model = criar_modelo(n_dyn, n_sta, horizon)
            model.load_state_dict(torch.load(modelo_path, map_location=device))
            model = model.to(device)
        else:
            # Criar e treinar novo modelo
            model = criar_modelo(n_dyn, n_sta, horizon)
            model, history = treinar_modelo(
                model, train_loader, val_loader,
                epochs=100, patience=10
            )
            
            # Salvar modelo
            torch.save(model.state_dict(), modelo_path)
            
            # Salvar histórico
            history_df = pd.DataFrame(history)
            history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
            history_df.to_parquet(history_path, index=False)

        # Avaliar no teste
        test_loss, test_mae, test_mse = avaliar_modelo(model, test_loader)
        
        # Salvar resultados do teste
        teste_df = pd.DataFrame([{
            'usina': usina,
            'loss': test_loss,
            'mae': test_mae,
            'mse': test_mse
        }])
        teste_df.to_parquet(teste_path, index=False)