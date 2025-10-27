# **Hybrid Hotel Recommendation System**

This project is a full-fledged, production-ready ML service that provides a hybrid hotel recommendation system. It leverages a user's social connections and personal preferences to deliver a personalized and relevant hotel ranking, designed to enhance user experience and increase conversion for travel platforms.

The core of the system is a **Deep & Cross Network with Residual Blocks (DCN-R)**, a sophisticated deep learning model designed to balance **memorization** of explicit user-item interactions and **generalization** to discover new, relevant recommendations.

## Core Architectural Concepts

The system is built on several key principles to ensure performance, relevance, and scalability.

*   **Two-Stage Recommendation Architecture:** To handle a large catalog of hotels efficiently, the system employs a two-stage process:
    1.  **Candidate Generation:** A fast and efficient selection of several hundred potentially relevant hotels from the entire database.
    2.  **Ranking:** A powerful, computationally intensive ML model that precisely sorts this smaller set of candidates in a personalized order for the user.

*   **Hybrid Candidate Generation:** The candidate pool is generated from multiple sources to ensure robustness and relevance in all scenarios:
    1.  **Social Signals:** Hotels highly rated by the user's friends form the core of the recommendations.
    2.  **Similarity Expansion:** The system uses learned item embeddings to find hotels that are similar to those liked by friends, aiding in discovery.
    3.  **Popularity Fallback:** For new users with no social connections or history, the system intelligently falls back to recommending popular and highly-rated hotels in the target city.

*   **Advanced Ranking Model (DCN-R):** The final ranking is performed by a custom DCN-R model, which excels at learning both explicit feature interactions (e.g., `city` + `stars`) and implicit, abstract user preferences.

*   **Containerized & Decoupled Architecture:** The entire application is containerized using Docker. The ML service and the database run as separate, communicating microservices, which is a standard for building scalable and maintainable systems.

## Tech Stack

*   **Machine Learning:** Python, PyTorch, Scikit-learn, Pandas, Optuna
*   **Backend Service:** FastAPI, SQLAlchemy
*   **Database:** PostgreSQL
*   **DevOps & Deployment:** Docker, Docker Compose

## System Architecture

The project is designed as a microservices-based system, consisting of two main components orchestrated by Docker Compose:

1.  **API Service:** A FastAPI application that contains all the ML logic. It loads the trained model and artifacts, and exposes RESTful endpoints for generating recommendations.
2.  **Database Service:** A PostgreSQL database that stores all core application data, including users, hotels, reviews, and friendships.

This decoupled design ensures that the ML inference is handled by a dedicated service that can be scaled independently.


## Getting Started

Follow these steps to run the complete project locally.

### Prerequisites

*   Git
*   Docker and Docker Compose

### 1. Clone the Repository

```bash
git clone [URL вашего репозитория]
cd [название папки]
```

### 2. Prepare Data and Artifacts

This repository contains the code and configuration, but not the large data files or trained model artifacts.

*   Create a directory named `data/` in the root of the project. Place `hackathon_augmented_data.csv` and `friendships.csv` inside it.
*   Create a directory named `artifacts/`. Place all pre-trained model files (`.pth`, `.gz`, `.npy`) inside it.

### 3. Configure the Environment

Copy the example environment file to create your local configuration.

```bash
cp .env.example .env
```
You do not need to edit this file to run the project locally, but you can change the database credentials if you wish.

### 4. Launch All Services

Build and run the API and database containers using Docker Compose.

```bash
docker-compose up --build -d
```
This command will build the Docker image for the API, start the PostgreSQL container in the background, and connect them.

### 5. Seed the Database

After the services have started, populate the database with the data from the CSV files. This is a one-time setup command.

```bash
docker-compose exec api python ml_training/database_setup.py
```
This command runs the database seeding script *inside* the already running API container.

### 6. Verification

The API is now running and fully operational.
*   The service is accessible at `http://localhost:8000`.
*   The interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

## API Usage

The service exposes two primary endpoints:

*   `POST /recommendations`: The main endpoint for fetching personalized recommendations. It requires `user_id`, `city`, and an optional `type` ('friends' or 'personal') in the request body.
*   `GET /similar_items`: A utility endpoint to find hotels similar to a given `item_id`.

For detailed request/response schemas and to try out the API live, please refer to the Swagger documentation at `/docs`.

## Model Documentation

For a deep dive into the model's architecture, the mathematics behind the Deep & Cross Network, the training methodology, and the results of our experimental validation (including ablation studies), please refer to the **`Documentation.md`** file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
