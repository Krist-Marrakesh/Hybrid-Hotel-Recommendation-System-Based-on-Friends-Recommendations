#Hybrid-Hotel-Recommendation-System-Based-on-Friends-Recommendations

This is a full-fledged ML service implementing a hybrid hotel recommendation system based on social connections and the user's personal preferences.

## Key Features

- **Two-Stage Architecture:** The system uses an effective "Candidate Selection + Ranking" approach for fast and accurate performance.
- **Hybrid Model:** Recommendations are based on multiple sources:
- **Friend Reviews:** Explicit recommendations from people in the user's social circle.
- **Similar Hotels:** Search for similar options using vector representations (item embeddings).
- **Personal Taste:** The final ranking takes into account the user's hidden preferences.
- **Cold start problem solved:** A fallback strategy featuring popular hotels is available for new users.
- **Ready-to-deploy API:** All logic is packaged in the FastAPI service and ready for integration.
- **Full containerization:** The project can be easily deployed with a single command thanks to Docker and Docker Compose.

## Tech Stack

- **ML & Data Science:** Python, PyTorch, Scikit-learn, Pandas, Optuna
- **Backend:** FastAPI, SQLAlchemy
- **Database:** PostgreSQL
- **Deployment:** Docker, Docker Compose

## How to Run the Project

1. **Clone the Repository:**
```bash
git clone [your repository URL]
cd ostrovok-recsys-portfolio
```

2. **Prepare Artifacts and Data:**
* Create a `data` folder and put `hackathon_augmented_data.csv` and `friendships.csv` in it.
* Create an `artifacts` folder and put all trained artifacts (`.pth`, `.gz`, `.npy`) in it.

3. **Create a .env` file:**
* Copy `.env.example` to a new `.env` file: `cp .env.example .env`.
* (Optional) Change passwords if necessary.

4. **Start all services:**
```bash
docker-compose up --build -d
```
This command will build the Docker image for the API, start the PostgreSQL container, and link them.

5. **Populate the database:**
* Run the script to populate the database (you need to be inside the API container).
```bash
docker-compose exec api python ml_training/database_setup.py
```

6. ** The service is accessible at `http://localhost:8000`. The interactive documentation (Swagger) is located at `http://localhost:8000/docs`.

## API Endpoints

- `POST /recommendations`: The main endpoint for receiving recommendations.
- `GET /similar_items`: Search for similar hotels.

For a detailed description of requests and responses, see the Swagger documentation at `/docs`.
