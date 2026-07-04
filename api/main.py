# Import Libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes.predict import router as predict_router
from api.routes.chat import router as chat_router
from api.routes.feedback import router as feedback_router
from deeplearning.inference import load_models

# Load Models in Singleton
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

# FastAPI
app = FastAPI(
    title='SEPHORA Skin Assistant',
    version='1.0.0',
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://srnurizki-aphrodia-ai.hf.space'],
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)

# Routers
app.include_router(predict_router, prefix='/predict', tags=['Predict'])
app.include_router(chat_router, prefix='/chat', tags=['Chat'])
app.include_router(feedback_router, prefix='/feedback', tags=['Feedback'])

# Health Status
@app.get('/health')
def health():
    return {'status': 'OK'}