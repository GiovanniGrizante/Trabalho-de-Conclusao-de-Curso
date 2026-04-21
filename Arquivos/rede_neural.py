import os, sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Previsão 1 - AMPL
# Previsão 2 - PINN

# Configurar dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preparar_dados(frame: pd.DataFrame, dyn_cols, static_cols, target_col, horizon):
    """
    Prepara dados para MLP.
    Cada amostra é um instante de tempo independente.
    
    Retorna:
        X_dyn: dados dinâmicos do instante t (não sequência)
        X_sta: dados estáticos do instante t
        y: target (emissão) - pode ser None se target_col não existir (PINN)
    """
    dyn = frame[dyn_cols].to_numpy(dtype="float32")
    sta = frame[static_cols].to_numpy(dtype="float32")
    
    # Verificar se a coluna target existe
    if target_col in frame.columns:
        tgt = frame[target_col].to_numpy(dtype="float32")
        tem_target = True
    else:
        tgt = None
        tem_target = False

    N = len(frame)
    X_dyn, X_sta, y = [], [], []
    
    for t in range(N - horizon + 1):
        X_dyn.append(dyn[t, :])
        X_sta.append(sta[t, :])
        if tem_target:
            y.append(tgt[t:t+horizon])
    
    if tem_target:
        return (np.stack(X_dyn, axis=0),
                np.stack(X_sta, axis=0),
                np.stack(y, axis=0))
    else:
        return (np.stack(X_dyn, axis=0),
                np.stack(X_sta, axis=0),
                None)

def criar_modelo(n_dyn, n_sta, horizon):
    """
    Cria um modelo MLP (Perceptron Multicamadas) para estimar emissões.
    Entrada: dados dinâmicos (instante t) + dados estáticos
    Saída: emissão prevista para horizonte h
    """
    class ModeloMLP(nn.Module):
        def __init__(self):
            super(ModeloMLP, self).__init__()
            
            input_dim = n_dyn + n_sta
            
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, horizon)
            )
            
        def forward(self, x_dyn, x_sta):
            # Concatena dinâmicas e estáticas
            x = torch.cat([x_dyn, x_sta], dim=1)
            return torch.relu(self.net(x)).squeeze()
    
    model = ModeloMLP()
    return model

def treinar_modelo( 
    previsao,
    model,
    train_loader,
    val_loader,
    dyn_cols,
    pg_mean,            # Para desnormalização
    pg_std,             # Para desnormalização
    emissoes_anuais,    # Para limitar a equação
    epochs=100,
    patience=50,
    huber_delta=0.5,
    peso_transicao=8.0,
    clip_grad=1.0,
    lambda_fis=100.0,
    lambda_anual = 100.0):
    """
    Treina o modelo MLP com Early Stopping e ReduceLR.
    Se previsao == '2', adiciona termo Physics-Informed na loss.
    """
    
    def previsao_1(model, train_loader, val_loader, dyn_cols,
    pg_mean, pg_std,
    epochs, patience, huber_delta,
    peso_transicao, clip_grad,
    lambda_fis):
        """
        Treinamento para previsao='1' (com dados sintéticos)
        """
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion_huber = nn.HuberLoss(delta=huber_delta)
        criterion_mae = nn.L1Loss()
        criterion_mse = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'loss': [], 'mae': [], 'mse': [], 'val_loss': [], 'val_mae': [], 'val_mse': []}

        for epoch in range(epochs):
            model.train()
            train_loss = train_mae = train_mse = 0.0

            for x_dyn, x_sta, y in train_loader:
                x_dyn, x_sta, y = x_dyn.to(device), x_sta.to(device), y.to(device)
                y = y.squeeze()
                optimizer.zero_grad()
                output = model(x_dyn, x_sta)

                loss_data = criterion_huber(output, y)
                is_transicao = x_sta[:, 0] == 1
                weight = torch.where(is_transicao, peso_transicao, 1.0)
                loss = (loss_data * weight).mean()  # ← APENAS loss_data

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()

                train_loss += loss.item()
                train_mae += criterion_mae(output, y).mean().item()
                train_mse += criterion_mse(output, y).mean().item()

            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            train_mse /= len(train_loader)

            model.eval()
            val_loss = val_mae = val_mse = 0.0
            
            # Validação do modelo treinado
            with torch.no_grad():
                for x_dyn, x_sta, y in val_loader:
                    x_dyn, x_sta, y = x_dyn.to(device), x_sta.to(device), y.to(device)
                    y = y.squeeze()
                    output = model(x_dyn, x_sta)
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

        if best_model_state:
            model.load_state_dict(best_model_state)
        return model, history
    
    def previsao_2(model, train_loader, val_loader, dyn_cols,
        pg_mean, pg_std, emissoes_anuais,
        epochs, patience, clip_grad,
        lambda_fis, lambda_anual):
        """
        Treinamento para previsao='2' (PINN com dados anuais)
        """
        model = model.to(device)
        pg_mean_t = torch.tensor(pg_mean, device=device)
        pg_std_t = torch.tensor(pg_std, device=device)

        alpha = nn.Parameter(torch.tensor(0.1, device=device))
        beta = nn.Parameter(torch.tensor(0.1, device=device))
        gamma = nn.Parameter(torch.tensor(0.01, device=device))
        omega = nn.Parameter(torch.tensor(0.05, device=device))
        mu = nn.Parameter(torch.tensor(1e-4, device=device))

        def emissao_fisica(PG):
            emissao = alpha + beta * PG + gamma * PG**2 + omega * torch.exp(mu * PG)
            return torch.relu(emissao)

        optimizer = optim.Adam(list(model.parameters()) + [alpha, beta, gamma, omega, mu], lr=1e-3)
        criterion_mse = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'loss': [], 'val_loss': []}
        idx_pg = dyn_cols.index("Geração")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for x_dyn, x_sta, anos in train_loader:
                x_dyn, x_sta, anos = x_dyn.to(device), x_sta.to(device), anos.to(device)
                optimizer.zero_grad()
                output = model(x_dyn, x_sta)
                output = output.squeeze()

                PG = x_dyn[:, idx_pg]
                PG_real = PG * pg_std_t + pg_mean_t
                emissao_fis = emissao_fisica(PG_real).squeeze()
                loss_fis = criterion_mse(output, emissao_fis)

                loss_anual = 0.0
                anos_unicos = torch.unique(anos)
                for ano in anos_unicos:
                    mask = anos == ano
                    soma_ano = output[mask].sum()
                    emissao_anual_real = emissoes_anuais[ano.item()]
                    loss_anual += (soma_ano - emissao_anual_real) ** 2
                loss_anual = loss_anual / len(anos_unicos)

                loss = lambda_fis * loss_fis + lambda_anual * loss_anual
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_dyn, x_sta in val_loader:
                    x_dyn, x_sta = x_dyn.to(device), x_sta.to(device)
                    output = model(x_dyn, x_sta)
                    PG = x_dyn[:, idx_pg]
                    PG_real = PG * pg_std_t + pg_mean_t
                    emissao_fis = emissao_fisica(PG_real)
                    loss_fis = criterion_mse(output, emissao_fis)
                    val_loss += loss_fis.item()

            val_loss /= len(val_loader)

            history['loss'].append(round(train_loss, 3))
            history['val_loss'].append(round(val_loss, 3))

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
        return model, history

    if previsao == '1':
        return previsao_1(model,
        train_loader,
        val_loader,
        dyn_cols,
        pg_mean,            # Para desnormalização
        pg_std,             # Para desnormalização
        epochs,
        patience,
        huber_delta,
        peso_transicao,
        clip_grad,
        lambda_fis)
    else:
        return previsao_2(model,
        train_loader,
        val_loader,
        dyn_cols,
        pg_mean,            # Para desnormalização
        pg_std,             # Para desnormalização
        emissoes_anuais,    # Para limitar a equação
        epochs,
        patience,
        clip_grad,
        lambda_fis,
        lambda_anual)

def avaliar_modelo(previsao, model, test_loader):
    def previsao_1(model, test_loader):
        """Avaliação para previsao='1' (com y_true)"""
        model.eval()
        criterion_mae = nn.L1Loss()
        criterion_mse = nn.MSELoss()
        test_loss = test_mae = test_mse = 0.0
        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for x_dyn, x_sta, y in test_loader:
                x_dyn, x_sta, y = x_dyn.to(device), x_sta.to(device), y.to(device)
                y = y.squeeze()
                output = model(x_dyn, x_sta)
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
    
    def previsao_2(model, test_loader):
        """Avaliação para previsao='2' (sem y_true)"""
        model.eval()
        y_pred_list = []

        with torch.no_grad():
            for x_dyn, x_sta in test_loader:
                x_dyn, x_sta = x_dyn.to(device), x_sta.to(device)
                output = model(x_dyn, x_sta)
                output = output.squeeze()
                y_pred_list.append(output.cpu().numpy())

        y_pred_all = np.concatenate(y_pred_list, axis=0)
        return None, None, None, None, y_pred_all

    if previsao == '1':
        return previsao_1(model, test_loader)
    else:
        return previsao_2(model, test_loader)

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
    # Funções para diferenciar os dados inseridos em cada análise
    
    def previsao_1(previsao):
        pasta = obter_iniciais('Qual modelo executar? (1 - Padrão | 2 - Regressão L2): ', [1, 2])
        pasta = 'Padrão' if pasta == 1 else 'Regressão L2'
        
        treinar = bool(obter_iniciais('Retreinar modelos? (1 - Sim | 0 - Não): ', [0, 1]))
        horizon = obter_iniciais('Qual a janela de previsão? (1h ou 24h): ', [1, 24])
        
        for usina in os.listdir('Usinas'):
            dir_dados = os.path.join('Usinas', usina, 'Minimização', pasta, 'Rede Neural', 'Dados')
            
            df_tr = pd.read_parquet(os.path.join(dir_dados, 'Treino.parquet'))
            df_val = pd.read_parquet(os.path.join(dir_dados, 'Validação.parquet'))
            df_te = pd.read_parquet(os.path.join(dir_dados, 'Teste.parquet'))
            
            target_col = 'Emissão'

            dyn_cols = (
                [c for c in df_tr.columns if c.startswith("Categoria de geração_")] +
                ["Geração", "Duração da Transição", "Fase da Transição",
                "Seno Índice", "Cosseno Índice"]
            )

            static_cols = (
                ["Potência Instalada", "Eficiência Energética [%]", "Fator de Capacidade [%]"] +
                [c for c in df_tr.columns if c.startswith("Combustível_")] +
                [c for c in df_tr.columns if c.startswith("Ciclo de Operação_")]
            )
            
            Xtr_dyn, Xtr_sta, ytr = preparar_dados(df_tr, dyn_cols, static_cols, target_col, horizon)
            Xva_dyn, Xva_sta, yva = preparar_dados(df_val, dyn_cols, static_cols, target_col, horizon)
            Xte_dyn, Xte_sta, yte = preparar_dados(df_te, dyn_cols, static_cols, target_col, horizon)

            n_dyn = Xtr_dyn.shape[1]
            n_sta = Xtr_sta.shape[1]
            horizon_real = ytr.shape[1]

            train_dataset = TensorDataset(
                torch.FloatTensor(Xtr_dyn),
                torch.FloatTensor(Xtr_sta),
                torch.FloatTensor(ytr)
            )
            
            val_dataset = TensorDataset(
                torch.FloatTensor(Xva_dyn),
                torch.FloatTensor(Xva_sta),
                torch.FloatTensor(yva)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(Xte_dyn),
                torch.FloatTensor(Xte_sta),
                torch.FloatTensor(yte)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            if Xtr_dyn.shape[0] == 0:
                continue
            
            modelo_dir = os.path.join('Usinas', usina, 'Minimização', pasta, 'Rede Neural', 'Modelos')
            historico_dir = os.path.join('Usinas', usina, 'Minimização', pasta, 'Rede Neural', 'Históricos')
            os.makedirs(modelo_dir, exist_ok=True)
            os.makedirs(historico_dir, exist_ok=True)

            modelo_path = os.path.join(modelo_dir, "best_model.pth")
            history_path = os.path.join(historico_dir, "Treino.parquet")
            teste_path = os.path.join(historico_dir, "Teste.parquet")

            if os.path.exists(modelo_path) and not treinar:
                model = criar_modelo(n_dyn, n_sta, horizon_real)
                model.load_state_dict(torch.load(modelo_path, map_location=device))
                model = model.to(device)
            else:
                model = criar_modelo(n_dyn, n_sta, horizon_real)
                model, history = treinar_modelo(previsao=previsao,
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    dyn_cols=dyn_cols, pg_mean=None, pg_std=None,
                    emissoes_anuais=None)
                
                torch.save(model.state_dict(), modelo_path)
                history_df = pd.DataFrame(history)
                history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
                history_df.to_parquet(history_path, index=False)

            test_loss, test_mae, test_mse, y_true_all, y_pred_all = avaliar_modelo(previsao, model, test_loader)

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

            teste_df = pd.DataFrame([{
                'usina': usina,
                'loss': round(test_loss, 3),
                'mae': round(test_mae, 3),
                'mse': round(test_mse, 3)
            }])
            teste_df.to_parquet(teste_path, index=False)
            
    def previsao_2(previsao):
        treinar = bool(obter_iniciais('Retreinar modelos? (1 - Sim | 0 - Não): ', [0, 1]))
        horizon = obter_iniciais('Qual a janela de previsão? (1h ou 24h): ', [1, 24])

        for usina in os.listdir('Usinas'):            
            dir_dados = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Dados')
            
            df_tr = pd.read_parquet(os.path.join(dir_dados, 'Treino.parquet'))
            df_val = pd.read_parquet(os.path.join(dir_dados, 'Validação.parquet'))
            df_te = pd.read_parquet(os.path.join(dir_dados, 'Teste.parquet'))
            df_scaler = pd.read_parquet(os.path.join(dir_dados, 'Scaler Geração.parquet'))
            
            emissoes_anuais = pd.read_parquet(os.path.join('Usinas', usina, 'Dados Externos', 'IEMA.parquet'))
            emissoes_anuais = dict(zip(emissoes_anuais['Ano'].astype(int), 
                                       emissoes_anuais['Emissões'].astype(float)))
            
            pg_mean = float(df_scaler.loc[0, 'Mean'])
            pg_std = float(df_scaler.loc[0, 'Std'])
            target_col = None  # Não há target horário

            dyn_cols = (
                [c for c in df_tr.columns if c.startswith("Categoria de geração_")] +
                ["Geração", "Duração da Transição", "Fase da Transição",
                "Seno Índice", "Cosseno Índice"]
            )

            static_cols = (
                ["Potência Instalada", "Eficiência Energética [%]", "Fator de Capacidade [%]"] +
                [c for c in df_tr.columns if c.startswith("Combustível_")] +
                [c for c in df_tr.columns if c.startswith("Ciclo de Operação_")]
            )
            
            Xtr_dyn, Xtr_sta, _ = preparar_dados(df_tr, dyn_cols, static_cols, target_col, horizon)
            Xva_dyn, Xva_sta, _ = preparar_dados(df_val, dyn_cols, static_cols, target_col, horizon)
            Xte_dyn, Xte_sta, _ = preparar_dados(df_te, dyn_cols, static_cols, target_col, horizon)
            
            ano_tr = df_tr['Ano'].copy().reset_index(drop=True)
            ano_tr = ano_tr.iloc[:len(Xtr_dyn)].values

            n_dyn = Xtr_dyn.shape[1]
            n_sta = Xtr_sta.shape[1]
            horizon_real = 1  # Para PINN, horizon=1 é recomendado

            modelo_dir = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Modelos')
            historico_dir = os.path.join('Usinas', usina, 'PGNM', 'Rede Neural', 'Históricos')
            os.makedirs(modelo_dir, exist_ok=True)
            os.makedirs(historico_dir, exist_ok=True)

            modelo_path = os.path.join(modelo_dir, "best_model.pth")
            history_path = os.path.join(historico_dir, "Treino.parquet")
            teste_path = os.path.join(historico_dir, "Teste.parquet")

            train_dataset = TensorDataset(
                torch.FloatTensor(Xtr_dyn),
                torch.FloatTensor(Xtr_sta),
                torch.LongTensor(ano_tr.astype(np.int64))
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(Xva_dyn),
                torch.FloatTensor(Xva_sta)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(Xte_dyn),
                torch.FloatTensor(Xte_sta)
            )

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            if Xtr_dyn.shape[0] == 0:
                continue

            if os.path.exists(modelo_path) and not treinar:
                model = criar_modelo(n_dyn, n_sta, horizon_real)
                model.load_state_dict(torch.load(modelo_path, map_location=device))
                model = model.to(device)
            else:
                model = criar_modelo(n_dyn, n_sta, horizon_real)
                model, history = treinar_modelo( previsao=previsao,
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    dyn_cols=dyn_cols, pg_mean=pg_mean, pg_std=pg_std,
                    emissoes_anuais=emissoes_anuais
                )
                torch.save(model.state_dict(), modelo_path)
                history_df = pd.DataFrame(history)
                history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
                history_df.to_parquet(history_path, index=False)

            _, _, _, _, y_pred_all = avaliar_modelo(previsao, model, test_loader)

            y_pred_flat = y_pred_all.flatten().astype(float)
            predicoes_df = pd.DataFrame({
                'Emissões Previstas': np.round(y_pred_flat, 3)
            })

            predicoes_path = os.path.join(historico_dir, "Predições.parquet")
            predicoes_df.to_parquet(predicoes_path, index=False)
    
    if previsao == '1':
        previsao_1(previsao)
    else:
        previsao_2(previsao)
