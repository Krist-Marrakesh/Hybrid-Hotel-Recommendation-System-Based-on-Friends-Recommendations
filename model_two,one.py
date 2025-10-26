import os
import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import sys
import random
import joblib
import logging
import plotly
import kaleido

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
logging.info(f"Device used:{device}")


def prepare_data(df, user_col, item_col, target_col, categorical_cols, numerical_cols):
    logging.info("1. Beginning data preprocessing and artifact creation...")

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

    logging.info("Preprocessing is complete.")
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

        # Get the number of ResNet blocks from the parameters
        n_res_blocks = params.get('n_res_blocks', 2) # Defaults to 2 if not specified

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n_cat, int(np.sqrt(n_cat)) + 1) for n_cat in cat_dims.values()])
        cat_emb_sum_dim = sum([int(np.sqrt(n_cat)) + 1 for n_cat in cat_dims.values()])
        input_dim = emb_dim * 2 + cat_emb_sum_dim + n_num_features

        self.initial_deep_layer = nn.Linear(input_dim, hidden_dim)

        #Create ResNet blocks in a loop and store them in ModuleList
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(n_res_blocks)
        ])

        self.cross_network = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross_layers)])

        final_dim = hidden_dim + input_dim
        self.final_linear = nn.Linear(final_dim, 1)

    def forward(self, user_ids, item_ids, cat_features, num_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        cat_embs = [emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_0 = torch.cat([user_emb, item_emb] + cat_embs + [num_features], dim=1)

        deep_out = self.initial_deep_layer(x_0)

        #Run the data through all ResNet blocks in a loop
        for res_block in self.res_blocks:
            deep_out = res_block(deep_out)

        cross_out = x_0
        for layer in self.cross_network: cross_out = layer(cross_out)
        final_input = torch.cat([deep_out, cross_out], dim=1)
        return self.final_linear(final_input).squeeze()


def objective(trial, train_data_tensors, val_data_tensors, model_dims):
    X_train_collab, X_train_cat, X_train_num, y_train = train_data_tensors
    X_val_collab, X_val_cat, X_val_num, y_val = val_data_tensors
    n_users, n_items, cat_dims, n_num_features = model_dims

    # HYPERPARAMETER SPACE
    params = {
        'emb_dim': trial.suggest_categorical("emb_dim", [16, 24, 32, 48, 64]),
        'hidden_dim': trial.suggest_int("hidden_dim", 32, 512, step=32),
        'n_cross_layers': trial.suggest_int("n_cross_layers", 1, 6),
        'n_res_blocks': trial.suggest_int("n_res_blocks", 1, 4),
        'dropout': trial.suggest_float("dropout", 0.1, 0.7, step=0.05),

        'lr': trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        'optimizer_name': trial.suggest_categorical("optimizer_name", ["AdamW", "Adam"]),

        'lr_scheduler_patience': trial.suggest_int("lr_scheduler_patience", 1, 3),
        'lr_scheduler_factor': trial.suggest_float("lr_scheduler_factor", 0.1, 0.5, step=0.1)
    }

    train_ds = TensorDataset(X_train_collab, X_train_cat, X_train_num, y_train)
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)

    # Create a model with new parameters
    model = DCN_RecSys(n_users, n_items, cat_dims, n_num_features, params).to(device)

    if params['optimizer_name'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    loss_fn = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=params['lr_scheduler_patience'],
        factor=params['lr_scheduler_factor']
    )

    # Training and validation cycle
    n_epochs, patience, best_val_loss, epochs_no_improve = 50, 5, float('inf'), 0
    for epoch in range(n_epochs):
        model.train()
        for X_collab_b, X_cat_b, X_num_b, y_b in train_dl:
            user_ids, item_ids = X_collab_b[:, 0].to(device), X_collab_b[:, 1].to(device)
            cat_features, num_features, y = X_cat_b.to(device), X_num_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(user_ids, item_ids, cat_features, num_features)
            loss = loss_fn(preds, y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            user_ids_val, item_ids_val = X_val_collab[:, 0].to(device), X_val_collab[:, 1].to(device)
            cat_features_val, num_features_val = X_val_cat.to(device), X_val_num.to(device)
            preds = model(user_ids_val, item_ids_val, cat_features_val, num_features_val)
            val_loss = loss_fn(preds, y_val.to(device).float()).item()

        scheduler.step(val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_path = f"checkpoints/best_model_trial_{trial.number}.pth"
            torch.save(model.state_dict(), model_path)
            trial.set_user_attr("best_model_path", model_path)
            logging.debug(
                f"Trial {trial.number}: New best val_loss: {val_loss:.4f}. Checkpoint saved in {model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1} for trial {trial.number}")
            break

    # AUC
    best_model_path = trial.user_attrs.get("best_model_path")
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            user_ids_val, item_ids_val = X_val_collab[:, 0].to(device), X_val_collab[:, 1].to(device)
            cat_features_val, num_features_val = X_val_cat.to(device), X_val_num.to(device)
            preds = model(user_ids_val, item_ids_val, cat_features_val, num_features_val)
            val_auc = roc_auc_score(y_val.cpu().numpy(), preds.cpu().numpy())
            trial.set_user_attr("AUC", val_auc)

    return best_val_loss


if __name__ == "__main__":
    # DATA PREPARATION
    try:
        df_full = pd.read_csv("data/hackathon_augmented_data.csv")
        ORIGINAL_USER_COL, ITEM_COL, TARGET_COL = 'guest_id', 'hotel_id', 'was_booked'

        df_full.rename(columns={ORIGINAL_USER_COL: 'user_id', ITEM_COL: 'item_id'}, inplace=True)
        USER_COL, ITEM_COL = 'user_id', 'item_id'

        logging.info(f"Dataset size before filtering: {len(df_full)}")
        df_full = df_full[(df_full['rating_overall'] >= 8) | (df_full['rating_overall'] <= 4)]
        logging.info(f"Dataset size after noise filtering: {len(df_full)}")

        logging.info("Creation of new features")
        df_full['price_per_star'] = (df_full['price_rub'] / df_full['stars']).replace([np.inf, -np.inf], 0).fillna(0)
        df_full['cleanliness_vs_service'] = (df_full['rating_cleanliness'] / df_full['rating_service']).replace(
            [np.inf, -np.inf], 0).fillna(0)
        df_full['location_premium'] = df_full['rating_overall'] - df_full['rating_location']
        logging.info("New features have been created.")

        CATEGORICAL_COLS = ['city', 'hotel_type']
        NUMERICAL_COLS = [
            'price_rub', 'stars', 'user_reviews_count', 'rating_overall', 'rating_location',
            'rating_cleanliness', 'rating_food', 'rating_service', 'price_per_star',
            'cleanliness_vs_service', 'location_premium'
        ]
    except FileNotFoundError:
        logging.error("Error: Data file not found.")
        sys.exit(1)

    train_tensors, val_tensors, model_dims, artifacts = prepare_data(df_full, USER_COL, ITEM_COL, TARGET_COL,
                                                                     CATEGORICAL_COLS, NUMERICAL_COLS)

    # LAUNCHING HYPERPARAMETRIC SEARCH
    os.makedirs("checkpoints", exist_ok=True)

    STUDY_NAME = "dcn_recsys_study_v2"
    STUDY_JOURNAL_FILE = f"{STUDY_NAME}.pkl"

    try:
        study = joblib.load(STUDY_JOURNAL_FILE)
        logging.info(f"Study '{STUDY_NAME}' loaded. {len(study.trials)} trials already completed.")
    except FileNotFoundError:
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), study_name=STUDY_NAME)
        logging.info(f"New study '{STUDY_NAME}' created.")
    try:
        study.optimize(lambda trial: objective(trial, train_tensors, val_tensors, model_dims), n_trials=300)
    except KeyboardInterrupt:
        logging.warning("Optimization interrupted by user.")
    finally:
        joblib.dump(study, STUDY_JOURNAL_FILE)
        logging.info(f"Study progress saved in {STUDY_JOURNAL_FILE}")

    if not study.best_trial:
        logging.error("No successful launches found. Terminating.")
        sys.exit(0)

    # VISUALIZATION
    print("\n" + "=" * 40)
    logging.info("SEARCH COMPLETED")
    best_trial = study.best_trial
    best_params = best_trial.params
    logging.info(f"Best trial: {best_trial.number}")
    logging.info(f"  - Best Validation LogLoss: {best_trial.value:.4f}")
    logging.info(f"  - Corresponding Validation AUC: {best_trial.user_attrs.get('AUC', 'N/A'):.4f}")
    logging.info(f"  - Best parameters: {best_params}")

    if optuna.visualization.is_available():
        logging.info("Creating graphs for analysis...")
        try:
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_image("optimization_history.png")

            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image("param_importances.png")

            logging.info("The graphs 'optimization_history.png' and 'param_importances.png' are saved.")
        except (ValueError, ImportError) as e:
            logging.warning(f"Failed to create charts: {e}. Please install plotly and kaleido.")
    else:
        logging.warning("To create graphs, install plotly: pip install plotly kaleido")

    # UPLOADING THE BEST MODEL AND FINAL EVALUATION
    logging.info("\n Loading the best model foun ")
    best_model_path = best_trial.user_attrs.get("best_model_path")
    if not best_model_path or not os.path.exists(best_model_path):
        logging.error(f"The best model path '{best_model_path}' was not found. Unable to continue.")
        sys.exit(1)

    n_users, n_items, cat_dims, n_num_features = model_dims
    final_model = DCN_RecSys(n_users, n_items, cat_dims, n_num_features, best_params).to(device)
    final_model.load_state_dict(torch.load(best_model_path))
    final_model.eval()
    logging.info(f"Best model successfully loaded from file: {best_model_path}")

    logging.info("\n Final evaluation of the model on validation data ")
    with torch.no_grad():
        (_, _, _, y_train) = train_tensors
        (X_val_collab, X_val_cat, X_val_num, y_val) = val_tensors

        user_ids_val, item_ids_val = X_val_collab[:, 0].to(device), X_val_collab[:, 1].to(device)
        cat_features_val, num_features_val = X_val_cat.to(device), X_val_num.to(device)

        preds_logits = final_model(user_ids_val, item_ids_val, cat_features_val, num_features_val)

        y_val_dev = y_val.to(device)
        final_val_logloss = nn.BCEWithLogitsLoss()(preds_logits, y_val_dev.float()).item()

        y_val_np = y_val.cpu().numpy()
        preds_logits_np = preds_logits.cpu().numpy()
        final_val_auc = roc_auc_score(y_val_np, preds_logits_np)

        preds_proba_np = torch.sigmoid(preds_logits).cpu().numpy()
        final_val_rmse = np.sqrt(mean_squared_error(y_val_np, preds_proba_np))

    logging.info(f"Final Validation LogLoss: {final_val_logloss:.4f}")
    logging.info(f"Final Validation AUC:     {final_val_auc:.4f}")
    logging.info(f"Final Validation RMSE:    {final_val_rmse:.4f}")

    # PRESERVATION OF FINAL ARTIFACTS
    logging.info("\n Saving final artifacts for the API")
    torch.save(final_model.state_dict(), "artifacts/final_dcn_model.pth")
    joblib.dump(artifacts, "artifacts/artifacts.gz")
    item_embeddings = final_model.item_embedding.weight.detach().cpu().numpy()
    np.save("artifacts/item_embeddings.npy", item_embeddings)
    joblib.dump(best_params, 'artifacts/best_params.gz')
    joblib.dump(model_dims, 'artifacts/model_dims.gz')
    logging.info("The final model and all artifacts are preserved.")
