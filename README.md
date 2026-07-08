# SkinSense (Aphrodia)

**AI-powered skincare recommendation system.** Aphrodia SkinSense analyzes a user's facial skin via CNN-based computer vision and recommends skincare products through a vectorless Retrieval-Augmented Generation (RAG) agent, deployed end-to-end on Google Cloud Platform.

![till](https://raw.githubusercontent.com/srnurizki/skinsense/master/demo.gif)
## Overview

Aphrodia SkinSense combines two AI subsystems:

1. **STACA AI** (Skin-Type-And-Concern-Analysis) — a computer vision pipeline that classifies a user's skin type and skin concerns from a selfie
2. **RAG Agent** — a conversational recommendation agent that queries a structured product database (not a vector store) to recommend skincare products matched to the user's skin profile and stated preferences

## Features

- Selfie-based skin type classification (oily / dry / normal)
- Selfie-based skin concern classification across 8 classes (e.g. wrinkles, pores, dark spots, redness, pigmentation)
- Per-facial-zone concern inference via MediaPipe Face Mesh
- Vectorless RAG product recommendation with natural-language query understanding
- Ingredient-to-concern content-based filtering to reduce hallucinated recommendations
- Automated model monitoring and retraining pipeline

## Model Performance

| Model | Task | Test Accuracy | Macro F1 |
|---|---|---|---|
| EfficientNetB3 | Skin type (3-class) | 90% | 0.86 |
| EfficientNetB4 | Skin concern (8-class) | 74% | 0.74 |

Dataset basis: 6+ sources including 2.8K+ skin & body care products, 1.09M+ product reviews, 3.7K+ facial images across skin type/concern classes, and 747 chemical ingredient records.

## Tech Stack

| Layer | Technology |
|---|---|
| CNN models | TensorFlow/Keras, EfficientNetB3, EfficientNetB4, MediaPipe Face Mesh |
| RAG / LLM | DeepSeek V4 Flash, LangGraph orchestration |
| Web search | SerpAPI |
| Backend API | FastAPI |
| Frontend | Streamlit (deployed on HuggingFace Spaces) |
| Database | PostgreSQL (Railway) |
| Object storage | Google Cloud Storage |
| Model registry | MLflow (GCS artifact store) |
| Containerization | Docker |
| Backend hosting | Google Cloud Run |
| CI/CD | GitHub Actions (Workload Identity Federation) |

## Architecture

```
User selfie/query
       |
       v
Streamlit UI (HuggingFace Spaces)
       |
       v
FastAPI backend (Cloud Run)
   |                      |
   v                      v
CNN Inference          RAG Agent (LangGraph + DeepSeek)
(EfficientNetB3/B4,       |
 MediaPipe)                v
   |                  PostgreSQL (Railway)
   v                  - product catalog
Skin profile          - user feedback
   |                      |
   `----------> merged into recommendation query
                          |
                          v
                  Ranked product results
                  (Wilson score, ingredient-concern match)
```

## MLOps Pipeline

- Model versions tracked in MLflow Model Registry with a GCS artifact store
- A monitoring script tracks incoming user feedback volume against a threshold
- Once the threshold is met, a GitHub Actions workflow (authenticated via Workload Identity Federation) triggers automated retraining and redeployment of the CNN models

## Project Structure

```
api/            FastAPI routes and request handling
agent/          RAG agent logic (LangGraph orchestration, DeepSeek integration)
deeplearning/   CNN model training, inference, and MediaPipe integration
storage/        PostgreSQL and Cloud Storage access layers
mlops/          Model monitoring and retraining scripts
ui/             Streamlit frontend
```

> Confirm this matches the current repo layout; reflects the last known project structure.

## Environment Variables

Key variables required at runtime (values excluded for security):

```
CONNECTION_STRING       # Railway PostgreSQL connection
MLFLOW_TRACKING_URI     # MLflow backend
DEEPSEEK_API_KEY
EXCHANGE_API_KEY
SERP_API_KEY
GITHUB_TOKEN
GOOGLE_CLOUD_PROJECT
```

## Local Development

```bash
# create and activate environment
conda create -n skinsense-env python=3.11
conda activate skinsense-env

# install dependencies
pip install -r requirements.txt

# set environment variables
cp .env.example .env

# run backend locally
uvicorn api.main:app --reload

# run frontend locally
streamlit run ui/app.py
```

## Deployment

Backend is containerized with Docker and deployed to Cloud Run. The frontend is deployed separately as a Streamlit app on HuggingFace Spaces. Cloud Run environment variables are configured via an `envvars.yaml` file rather than inline flags, for reliability with special-character values.

## Known Limitations

- Dark spots classification underperforms other skin concern classes due to visual overlap with inflammatory acne
- Skin type and concern accuracy (90% / 74%) reflect realistic benchmarks for selfie-based classification, not clinical-grade diagnostics
- Recommendations are constrained to the product catalog available in the PostgreSQL database at inference time

## Author

Satryo Akbar Nurizki - [GitHub](https://github.com/srnurizki)

## Acknowledgements

Built as a capstone project for Dibimbing.id's "From Problem to Production" Data Science & Machine Learning bootcamp.
