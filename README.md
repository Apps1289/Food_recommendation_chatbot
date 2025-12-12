# Food Recommendation Chatbot

> A conversational chatbot that recommends food/dishes and restaurants based on user preferences, dietary restrictions, context, and past interactions. Built to be extended for different recommendation strategies (content-based, collaborative filtering, hybrid) and easily deployable as an API or web app.

[![Python][badge-python]][python-url] [![License][badge-license]][license-url] [![Repo size][badge-size]][repo-url]

Table of contents
- [Features](#features)
- [Architecture / Components](#architecture--components)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data format](#data-format)
- [Training & Evaluation](#training--evaluation)
- [Running the chatbot (local)](#running-the-chatbot-local)
- [API / Endpoints](#api--endpoints)
- [Deployment (Docker)](#deployment-docker)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

Features
- Conversational interface for food and restaurant recommendations.
- Support for preference and constraint handling (price, cuisine, dietary restrictions).
- Extensible recommendation engine (content-based, collaborative, hybrid).
- Simple training pipeline for model-based recommenders.
- REST API for integration with web or mobile frontends.
- Logging of user sessions for personalization and offline training.

Architecture / Components
- frontend/ — (optional) web UI or demo chat client.
- api/ — REST API (FastAPI/Flask) exposing chat and recommendation endpoints.
- models/ — recommendation models, training scripts, saved checkpoints.
- data/ — datasets (CSV/JSON), schema and sample data.
- utils/ — preprocessing, tokenization, embedding utilities, helpers.
- requirements.txt — Python dependencies.
- Dockerfile / docker-compose.yml — for containerized deployment.

Quickstart (recommended)
1. Clone the repo:
   git clone https://github.com/Apps1289/Food_recommendation_chatbot.git
2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
3. Prepare data (see [Data format](#data-format)).
4. Train the recommendation model (if applicable):
   python models/train.py --data data/your_dataset.csv --out models/checkpoint.pkl
5. Start the API:
   uvicorn api.main:app --reload
6. Interact with the chatbot using the web UI (if provided) or via the API.

Installation
- Requirements
  - Python 3.8+
  - pip
  - (Optional) Docker
- Install dependencies:
  pip install -r requirements.txt

Configuration
- Environment variables (example)
  - APP_ENV=development
  - DATABASE_URL=sqlite:///data/db.sqlite3
  - MODEL_PATH=models/checkpoint.pkl
  - LOG_LEVEL=info
- Example .env
  APP_ENV=development
  MODEL_PATH=models/checkpoint.pkl
  DATABASE_URL=sqlite:///data/db.sqlite3

Data format
- Typical dataset (CSV/JSON) columns:
  - user_id (optional)
  - item_id (dish_id / restaurant_id)
  - item_name
  - cuisine
  - price_level (e.g., 1-4)
  - dietary_tags (e.g., vegetarian, vegan, gluten-free)
  - rating (optional)
  - timestamp (optional)
- Example CSV row:
  user_id,item_id,item_name,cuisine,price_level,dietary_tags,rating,timestamp
  123,987,"Margherita Pizza","Italian",2,"vegetarian",4.5,2024-05-12T18:30:00Z

Training & Evaluation
- Train the recommender:
  python models/train.py --data data/ratings.csv --model output/recommender.pkl --epochs 10
- Common flags:
  --model-type [content|collab|hybrid]
  --embedding-dim 64
  --batch-size 256
  --learning-rate 0.001
- Evaluation metrics:
  - RMSE / MAE (for rating prediction)
  - Precision@K, Recall@K, NDCG (for top-K recommendations)
  - A/B testing for online evaluations
- Saved artifacts:
  - model checkpoint files (models/)
  - vectorizer/encoder artifacts
  - training logs (logs/)

Running the chatbot (local)
- Start API server:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
- Example request (curl):
  curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "u123", "message": "I want a vegan dinner under $20"}'
- Expected response (example):
  {
    "reply": "Here are some vegan options near you: ...",
    "recommendations": [
      {"id": "r1", "name": "Vegan Delight", "cuisine": "Vegan", "price_level": 2}
    ],
    "session_id": "s_abc123"
  }

API / Endpoints (example)
- POST /chat
  - body: { user_id, message, context? }
  - returns: assistant reply + recommendations
- GET /recommendations?user_id=...&k=5
  - returns: list of k recommended items
- POST /feedback
  - body: { user_id, item_id, feedback_type, value }
  - Use for logging explicit feedback and improving recommendations

Deployment (Docker)
- Build:
  docker build -t food-recommender:latest .
- Run:
  docker run -e MODEL_PATH=/app/models/checkpoint.pkl -p 8000:8000 food-recommender:latest
- docker-compose (example):
  version: "3.8"
  services:
    app:
      build: .
      ports:
        - "8000:8000"
      environment:
        - MODEL_PATH=/app/models/checkpoint.pkl

Logging, Privacy & Data Handling
- Persist only what is necessary for recommendations and debugging.
- Anonymize PII when storing logs.
- If collecting user data, ensure compliance with applicable privacy laws and provide opt-outs.

Extending the project
- Swap recommendation strategies: implement a new class in models/ and add a CLI flag to train.py.
- Add slots/intents parsing for richer conversations: use Rasa, spaCy or a transformer-based intent classifier.
- Add personalization: store user profiles and interaction history and incorporate into scoring.

Contributing
- Fork the repo and open a PR.
- Follow the repo's code style and add tests for new functionality.
- Describe changes clearly and link related issues.
- If adding a dataset, include a BALANCE.md note and sample-only small dataset or provide scripts to download public datasets.

Troubleshooting
- Dependencies failing to install: ensure Python version and OS toolchain available (e.g., build-essential).
- Model not loading: verify MODEL_PATH environment variable and artifact existence.
- API errors: check logs/ and start server in debug mode.

License
This project is provided under the MIT License. See LICENSE.md for details.

Contact
Maintainer: Apps1289
Repo: https://github.com/Apps1289/Food_recommendation_chatbot

Notes / TODO
- Add example dataset and minimal demo UI in frontend/.
- Provide ready-to-run Docker Compose with a simple SQLite store and demo client.
- Add CI status badge and test coverage badge.

<!-- Badge placeholders -->
[badge-python]: https://img.shields.io/badge/python-3.8%2B-blue
[badge-license]: https://img.shields.io/badge/license-MIT-green
[badge-size]: https://img.shields.io/github/repo-size/Apps1289/Food_recommendation_chatbot
[python-url]: https://www.python.org/
[license-url]: https://opensource.org/licenses/MIT
[repo-url]: https://github.com/Apps1289/Food_recommendation_chatbot
