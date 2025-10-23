import sys
import random
import traceback
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

print("Запуск ML-сервиса на FastAPI...")


# PYDANTIC MODELS


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., example=15, description="ID пользователя для персонализации")
    city: str = Field(..., example="Сочи", description="Город, в котором ищутся отели")
    type: str = Field("friends", example="personal", description="Тип рекомендаций: 'friends' или 'personal'")


class SimilarItemsResponse(BaseModel):
    similar_item_ids: List[int]


class HotelResponse(BaseModel):
    hotel_id: int
    city: str | None
    price_rub: float | None
    stars: float | None
    recommended_by: List[int]


class RecommendationResponse(BaseModel):
    ranked_hotels: List[HotelResponse]
    message: str | None = None


# MODEL CLASSES

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = x.unsqueeze(2)
        x_t = x.unsqueeze(1)
        return x_0.squeeze(2) + torch.matmul(x_0, self.w(x_t)).squeeze(2) + self.b


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.layer1(x)))
        out = self.dropout(out)
        out = self.bn2(self.layer2(out))
        out += identity
        out = self.relu(out)
        return out


class DCN_RecSys(nn.Module):
    def __init__(self, n_users: int, n_items: int, cat_dims: Dict[str, int], n_num_features: int,
                 params: Dict[str, Any]):
        super().__init__()
        emb_dim = params['emb_dim']
        hidden_dim = params['hidden_dim']
        n_cross_layers = params['n_cross_layers']
        dropout = params['dropout']

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, int(np.sqrt(n_cat)) + 1) for n_cat in cat_dims.values()])

        cat_emb_sum_dim = sum([int(np.sqrt(n_cat)) + 1 for n_cat in cat_dims.values()])
        input_dim = emb_dim * 2 + cat_emb_sum_dim + n_num_features

        # Deep-part
        self.initial_deep_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResBlock(hidden_dim, dropout)
        self.res_block2 = ResBlock(hidden_dim, dropout)

       # Cross-part
        self.cross_network = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross_layers)])

        # Final layer
        final_dim = hidden_dim + input_dim
        self.final_linear = nn.Linear(final_dim, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, cat_features: torch.Tensor,
                num_features: torch.Tensor) -> torch.Tensor:
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        cat_embs = [emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)]

        # Formation of the input vector x0
        x_0 = torch.cat([user_emb, item_emb] + cat_embs + [num_features], dim=1)

        # Run through Deep & Cross parts
        deep_out = self.res_block2(self.res_block1(self.initial_deep_layer(x_0)))
        cross_out = x_0
        for layer in self.cross_network:
            cross_out = layer(cross_out)

       # Concatenation and final prediction
        final_input = torch.cat([deep_out, cross_out], dim=1)
        return self.final_linear(final_input).squeeze()


# GLOBAL ARTIFACT STORAGE
ml_artifacts: Dict[str, Any] = {}
app = FastAPI(title="Ostrovok Hybrid Recommendation API", version="3.0-clean")


#ASSISTANT FUNCTIONS

def get_friends_for_user(user_id: int) -> set:
    friendships_df = ml_artifacts.get('friendships_df')
    if friendships_df is None or friendships_df.empty:
        return set()
    part1 = friendships_df[friendships_df['user_id_1'] == user_id]['user_id_2'].tolist()
    part2 = friendships_df[friendships_df['user_id_2'] == user_id]['user_id_1'].tolist()
    return set(part1 + part2)


def _generate_candidates(user_id: int, target_city: str, mode: str) -> set:
    main_df = ml_artifacts['main_df']
    negative_filter = set()
    positive_recs = []

    if mode == 'friends':
        source_ids = get_friends_for_user(user_id)
        source_reviews = main_df[main_df['user_id'].isin(source_ids)] if source_ids else pd.DataFrame()
    else:  # mode == 'personal'
        source_reviews = main_df[main_df['user_id'] == user_id]

    if not source_reviews.empty:
        positive_recs = source_reviews[source_reviews['rating_overall'] >= 8]['item_id'].unique().tolist()
        negative_filter = set(source_reviews[source_reviews['rating_overall'] <= 4]['item_id'].unique())

    candidate_hotels = set(positive_recs)

    # Expansion through similarity
    for hotel_id in positive_recs:
        internal_id = ml_artifacts['artifacts']['item_id_mapping'].get(hotel_id)
        if internal_id is not None:
            item_vector = ml_artifacts['item_embeddings'][internal_id].reshape(1, -1)
            _, indices = ml_artifacts['nn_model'].kneighbors(item_vector, n_neighbors=11)
            neighbor_ids = [ml_artifacts['reverse_item_map'][idx] for idx in indices.squeeze()[1:] if
                            idx in ml_artifacts['reverse_item_map']]
            candidate_hotels.update(neighbor_ids)

   # Fallback strategy
    if len(candidate_hotels) < 20:
        popular_hotels = \
        main_df[main_df['city'] == target_city].sort_values(by='user_reviews_count', ascending=False).head(100)[
            'item_id'].tolist()
        candidate_hotels.update(popular_hotels)

    # Final filtration
    city_hotels = set(main_df[main_df['city'] == target_city]['item_id'].unique())
    candidate_hotels.intersection_update(city_hotels)
    candidate_hotels.difference_update(negative_filter)

    return candidate_hotels


def preprocess_for_ranking(items_df: pd.DataFrame, user_id: int) -> Tuple[torch.Tensor, ...]:
    artifacts = ml_artifacts['artifacts']
    internal_user_id = artifacts['user_id_mapping'].get(user_id, len(artifacts['user_id_mapping']) // 2)

    collab_data = items_df[['item_id']].copy()
    collab_data['user_id'] = internal_user_id
    collab_data['item_id_encoded'] = collab_data['item_id'].map(artifacts['item_id_mapping']).fillna(0)

    X_collab_user = torch.tensor(collab_data['user_id'].values, dtype=torch.long)
    X_collab_item = torch.tensor(collab_data['item_id_encoded'].values, dtype=torch.long)

    cat_data = pd.DataFrame()
    for col, encoder in artifacts['cat_encoders'].items():
        cat_data[f'{col}_encoded'] = items_df[col].map(encoder).fillna(0)
    X_cat = torch.tensor(cat_data.values, dtype=torch.long)

    numerical_cols = artifacts['numerical_cols']
    num_data_scaled = artifacts['scaler'].transform(items_df[numerical_cols])
    X_num = torch.tensor(num_data_scaled, dtype=torch.float32)

    return X_collab_user, X_collab_item, X_cat, X_num



@app.on_event("startup")
def load_artifacts():
    print("INFO:     Загрузка артефактов при старте сервера...")
    try:
        device = torch.device("cpu")
        ml_artifacts['device'] = device

        # Loading data (In-Memory DB)
        main_df_raw = pd.read_csv("hackathon_augmented_data.csv")
        main_df_raw.rename(columns={'guest_id': 'user_id', 'hotel_id': 'item_id'}, inplace=True)
        ml_artifacts['main_df'] = main_df_raw
        ml_artifacts['friendships_df'] = pd.read_csv("friendships.csv")

        # Recreating features generated before training
        print("INFO:     Воссоздание признаков для инференса...")
        main_df_raw['price_per_star'] = (main_df_raw['price_rub'] / main_df_raw['stars']).replace([np.inf, -np.inf],
                                                                                                  0).fillna(0)
        main_df_raw['cleanliness_vs_service'] = (
                    main_df_raw['rating_cleanliness'] / main_df_raw['rating_service']).replace([np.inf, -np.inf],
                                                                                               0).fillna(0)
        main_df_raw['location_premium'] = main_df_raw['rating_overall'] - main_df_raw['rating_location']

        # Loading ML artifacts
        print("INFO:     Загрузка ML артефактов (модели, скейлеры, энкодеры)...")
        ml_artifacts['artifacts'] = joblib.load("artifacts.gz")
        model_dims = joblib.load("model_dims.gz")
        best_params = joblib.load("best_params.gz")
        ml_artifacts['item_embeddings'] = np.load("item_embeddings.npy")

        # Инициализация и загрузка весов DCN-модели
        n_users, n_items, cat_dims, n_num_features = model_dims
        model = DCN_RecSys(n_users, n_items, cat_dims, n_num_features, best_params)
        model.load_state_dict(torch.load("final_dcn_model.pth", map_location=device))
        model.to(device)
        model.eval()
        ml_artifacts['final_model'] = model

        # Initialization and loading of DCN model weights
        nn_model = NearestNeighbors(n_neighbors=16, metric='cosine', algorithm='brute')
        nn_model.fit(ml_artifacts['item_embeddings'])
        ml_artifacts['nn_model'] = nn_model

        ml_artifacts['reverse_item_map'] = {v: k for k, v in ml_artifacts['artifacts']['item_id_mapping'].items()}
        print("INFO:     Артефакты успешно загружены. Сервер готов.")

    except Exception as e:
        print(f"CRITICAL: КРИТИЧЕСКАЯ ОШИБКА при запуске: {e}")
        traceback.print_exc()
        sys.exit(1)


# API
@app.get("/similar_items", response_model=SimilarItemsResponse)
def get_similar_items_endpoint(item_id: int = Query(..., example=123), n: int = Query(10, ge=1, le=50)):
    internal_id = ml_artifacts['artifacts']['item_id_mapping'].get(item_id)
    if internal_id is None:
        raise HTTPException(status_code=404, detail=f"Отель с ID {item_id} не найден.")

    item_vector = ml_artifacts['item_embeddings'][internal_id].reshape(1, -1)
    _, indices = ml_artifacts['nn_model'].kneighbors(item_vector, n_neighbors=n + 1)
    neighbor_ids = [ml_artifacts['reverse_item_map'][idx] for idx in indices.squeeze()[1:] if
                    idx in ml_artifacts['reverse_item_map']]

    return {"similar_item_ids": neighbor_ids}


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations_endpoint(request_data: RecommendationRequest):
    try:
        candidate_hotels = _generate_candidates(request_data.user_id, request_data.city, request_data.type)
        if not candidate_hotels:
            return {"ranked_hotels": [], "message": "Подходящих кандидатов не найдено."}

        main_df = ml_artifacts['main_df']
        items_to_rank_df = main_df[main_df['item_id'].isin(list(candidate_hotels))].drop_duplicates(subset=['item_id'])
        if items_to_rank_df.empty:
            return {"ranked_hotels": [], "message": "Нет данных по найденным отелям для ранжирования."}

        X_user, X_item, X_cat, X_num = preprocess_for_ranking(items_to_rank_df, request_data.user_id)
        with torch.no_grad():
            preds = ml_artifacts['final_model'](X_user, X_item, X_cat, X_num)

        scores = preds.cpu().numpy()
        scored_items = sorted(zip(scores, items_to_rank_df['item_id']), key=lambda x: x[0], reverse=True)
        ranked_ids = [item_id for score, item_id in scored_items]

        rich_info = main_df[main_df['item_id'].isin(ranked_ids)].drop_duplicates(subset=['item_id']).set_index(
            'item_id').to_dict('index')

        friends_ids = get_friends_for_user(request_data.user_id)
        friends_reviews = main_df[main_df['user_id'].isin(friends_ids)] if friends_ids else pd.DataFrame()
        pos_rec_map = {}
        if not friends_reviews.empty:
            pos_rec_map = friends_reviews[friends_reviews['rating_overall'] >= 8].groupby('item_id')[
                'user_id'].unique().apply(list).to_dict()

        final_response = []
        for hotel_id in ranked_ids:
            hotel_data = rich_info.get(hotel_id, {})
            final_response.append({
                "hotel_id": hotel_id,
                "city": hotel_data.get("city"),
                "price_rub": hotel_data.get("price_rub"),
                "stars": hotel_data.get("stars"),
                "recommended_by": pos_rec_map.get(hotel_id, [])
            })

        return {"ranked_hotels": final_response}

    except Exception as e:
        print(f"CRITICAL: Ошибка во время выполнения запроса /recommendations: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера.")


if __name__ == '__main__':

    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)
