import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import random
import joblib


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


def prepare_data(df, user_col, item_col, target_col, categorical_cols, numerical_cols):
    print("1. Начало предобработки данных и создания артефактов...")

    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df.dropna(subset=categorical_cols, inplace=True)

    user_map = {original: i for i, original in enumerate(df[user_col].unique())}
    item_map = {original: i for i, original in enumerate(df[item_col].unique())}
    df['user_id_encoded'] = df[user_col].map(user_map)
    df['item_id_encoded'] = df[item_col].map(item_map)

    cat_encoders = {}
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        cat_encoders[col] = {cat: i for i, cat in enumerate(df[col].cat.categories)}
        df[f'{col}_encoded'] = df[col].cat.codes

    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    n_users, n_items = len(user_map), len(item_map)
    cat_dims = {col: len(enc) for col, enc in cat_encoders.items()}
    n_num_features = len(numerical_cols)
    model_dims = (n_users, n_items, cat_dims, n_num_features)


    X_collab = df[['user_id_encoded', 'item_id_encoded']].values
    X_cat = df[[f'{col}_encoded' for col in categorical_cols]].values
    X_num = df[numerical_cols].values
    y = df[target_col].values.astype(np.float32)

    indices = np.arange(len(df))
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(indices, y, test_size=0.2, random_state=42)

    train_tensors = (
        torch.tensor(X_collab[X_train_idx], dtype=torch.long),
        torch.tensor(X_cat[X_train_idx], dtype=torch.long),
        torch.tensor(X_num[X_train_idx], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32))
    val_tensors = (
        torch.tensor(X_collab[X_val_idx], dtype=torch.long),
        torch.tensor(X_cat[X_val_idx], dtype=torch.long),
        torch.tensor(X_num[X_val_idx], dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32))

    artifacts = {
        "user_id_mapping": user_map, "item_id_mapping": item_map,
        "scaler": scaler, "cat_encoders": cat_encoders,
        "numerical_cols": numerical_cols, "categorical_cols": categorical_cols
    }

    print("Предобработка завершена.")
    return train_tensors, val_tensors, model_dims, artifacts


class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        x_0 = x.unsqueeze(2)
        x_t = x.unsqueeze(1)
        return x_0.squeeze(2) + torch.matmul(x_0, self.w(x_t)).squeeze(2) + self.b


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class DCN_RecSys(nn.Module):
    def __init__(self, n_users, n_items, cat_dims, n_num_features, params):
        super().__init__()
        emb_dim = params['emb_dim']
        hidden_dim = params['hidden_dim']
        n_cross_layers = params['n_cross_layers']
        dropout = params['dropout']

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n_cat, int(np.sqrt(n_cat)) + 1) for n_cat in cat_dims.values()])
        cat_emb_sum_dim = sum([int(np.sqrt(n_cat)) + 1 for n_cat in cat_dims.values()])
        input_dim = emb_dim * 2 + cat_emb_sum_dim + n_num_features

        self.initial_deep_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResBlock(hidden_dim, dropout)
        self.res_block2 = ResBlock(hidden_dim, dropout)

        self.cross_network = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross_layers)])

        final_dim = hidden_dim + input_dim
        self.final_linear = nn.Linear(final_dim, 1)

    def forward(self, user_ids, item_ids, cat_features, num_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        cat_embs = [emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_0 = torch.cat([user_emb, item_emb] + cat_embs + [num_features], dim=1)

        deep_out = self.initial_deep_layer(x_0)
        deep_out = self.res_block1(deep_out)
        deep_out = self.res_block2(deep_out)

        cross_out = x_0
        for layer in self.cross_network: cross_out = layer(cross_out)
        final_input = torch.cat([deep_out, cross_out], dim=1)
        return self.final_linear(final_input).squeeze()


def objective(trial, train_data_tensors, val_data_tensors, model_dims):
    X_train_collab, X_train_cat, X_train_num, y_train = train_data_tensors
    X_val_collab, X_val_cat, X_val_num, y_val = val_data_tensors
    n_users, n_items, cat_dims, n_num_features = model_dims

    params = {
        'emb_dim': trial.suggest_int("emb_dim", 16, 64, step=8),
        'hidden_dim': trial.suggest_int("hidden_dim", 64, 256, step=32),
        'n_cross_layers': trial.suggest_int("n_cross_layers", 2, 6),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
        'dropout': trial.suggest_float("dropout", 0.2, 0.7, step=0.05),
        'weight_decay': trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    }

    train_ds = TensorDataset(X_train_collab, X_train_cat, X_train_num, y_train)
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    model = DCN_RecSys(n_users, n_items, cat_dims, n_num_features, params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    n_epochs, patience, best_val_loss, epochs_no_improve = 50, 5, float('inf'), 0
    for epoch in range(n_epochs):
        model.train()
        for X_collab_b, X_cat_b, X_num_b, y_b in train_dl:
            user_ids, item_ids = X_collab_b[:, 0].to(device), X_collab_b[:, 1].to(device)
            cat_features, num_features, y = X_cat_b.to(device), X_num_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(user_ids, item_ids, cat_features, num_features)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            user_ids_val, item_ids_val = X_val_collab[:, 0].to(device), X_val_collab[:, 1].to(device)
            cat_features_val, num_features_val = X_val_cat.to(device), X_val_num.to(device)
            preds = model(user_ids_val, item_ids_val, cat_features_val, num_features_val)
            val_loss = loss_fn(preds, y_val.to(device)).item()

        scheduler.step(val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience: break

    return best_val_loss


if __name__ == "__main__":
    try:
        df_full = pd.read_csv("hackathon_augmented_data.csv")
        ORIGINAL_USER_COL, ITEM_COL, TARGET_COL = 'guest_id', 'hotel_id', 'was_booked'

        df_full.rename(columns={ORIGINAL_USER_COL: 'user_id', ITEM_COL: 'item_id'}, inplace=True)
        USER_COL, ITEM_COL = 'user_id', 'item_id'

        print(f"Размер датасета до фильтрации: {len(df_full)}")
        df_full = df_full[(df_full['rating_overall'] >= 8) | (df_full['rating_overall'] <= 4)]
        print(f"Размер датасета после фильтрации шума: {len(df_full)}")
        print("Создание новых признаков...")
        df_full['price_per_star'] = (df_full['price_rub'] / df_full['stars']).replace([np.inf, -np.inf], 0).fillna(0)
        df_full['cleanliness_vs_service'] = (df_full['rating_cleanliness'] / df_full['rating_service']).replace(
            [np.inf, -np.inf], 0).fillna(0)
        df_full['location_premium'] = df_full['rating_overall'] - df_full['rating_location']
        print("Новые признаки созданы.")

        CATEGORICAL_COLS = ['city', 'hotel_type']
        NUMERICAL_COLS = [
            'price_rub', 'stars', 'user_reviews_count',
            'rating_overall', 'rating_location', 'rating_cleanliness',
            'rating_food', 'rating_service',
            'price_per_star', 'cleanliness_vs_service', 'location_premium'
        ]

    except FileNotFoundError:
        print("Ошибка: Файл 'hackathon_rich_data.csv' не найден.")
        sys.exit(1)

    train_tensors, val_tensors, model_dims, artifacts = prepare_data(df_full, USER_COL, ITEM_COL, TARGET_COL,
                                                                     CATEGORICAL_COLS, NUMERICAL_COLS)


    func = lambda trial: objective(trial, train_tensors, val_tensors, model_dims)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(func, n_trials=100)

    if not study.best_trial:
        print("Успешных запусков не найдено. Завершение работы.");
        sys.exit(0)

    print("\n--- ПОИСК ЗАВЕРШЕН ---")
    best_params = study.best_params
    print(f"Лучший Validation RMSE: {np.sqrt(study.best_value)}")
    print("Лучшие параметры:", best_params)


    print("\n--- Переобучение лучшей модели ---")
    (X_train_collab, X_train_cat, X_train_num, y_train) = train_tensors
    (X_val_collab, X_val_cat, X_val_num, y_val) = val_tensors

    X_full_collab = torch.cat([X_train_collab, X_val_collab])
    X_full_cat = torch.cat([X_train_cat, X_val_cat])
    X_full_num = torch.cat([X_train_num, X_val_num])
    y_full = torch.cat([y_train, y_val])

    full_dataset = TensorDataset(X_full_collab, X_full_cat, X_full_num, y_full)
    full_loader = DataLoader(full_dataset, batch_size=best_params['batch_size'], shuffle=True)

    n_users, n_items, cat_dims, n_num_features = model_dims
    final_model = DCN_RecSys(n_users, n_items, cat_dims, n_num_features, best_params).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['lr'],
                                  weight_decay=best_params.get('weight_decay', 1e-5))
    loss_fn = nn.MSELoss()

    FINAL_EPOCHS = 40
    print(f"Обучение на {FINAL_EPOCHS} эпох...")
    for epoch in range(FINAL_EPOCHS):
        final_model.train()
        epoch_loss = 0
        for X_collab_b, X_cat_b, X_num_b, y_b in full_loader:
            user_ids, item_ids = X_collab_b[:, 0].to(device), X_collab_b[:, 1].to(device)
            cat_features, num_features, y = X_cat_b.to(device), X_num_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = final_model(user_ids, item_ids, cat_features, num_features)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)
        avg_loss = epoch_loss / len(full_dataset)
        print(f"Эпоха {epoch + 1}/{FINAL_EPOCHS}, Loss: {avg_loss:.4f}, RMSE: {np.sqrt(avg_loss):.4f}")


    print("\n--- Сохранение артефактов для API ---")
    torch.save(final_model.state_dict(), "final_dcn_model.pth")
    joblib.dump(artifacts, "artifacts.gz")
    item_embeddings = final_model.item_embedding.weight.detach().cpu().numpy()
    np.save("item_embeddings.npy", item_embeddings)
    joblib.dump(best_params, 'best_params.gz')
    joblib.dump(model_dims, 'model_dims.gz')
    print("Финальная модель и все артефакты сохранены. Готово для интеграции с бэкендом.")