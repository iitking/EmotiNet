import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import BASE_DIR, EMOTION_LABELS, EMOTION_DETAILS, MODEL_BENCHMARKS
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)
from app.model import classifier

# Initialize FastAPI App with comprehensive metadata
app = FastAPI(
    title="EmotiNet - Neural Emotion Recognition API",
    description="""
    🚀 **EmotiNet** is a state-of-the-art Deep Learning system for real-time text emotion detection.
    Powered by a **Bidirectional GRU (BiGRU)** neural network trained on the `dair-ai/emotion` benchmark dataset.
    
    ### Detected Emotions:
    * 😊 **Joy**
    * 😢 **Sadness**
    * 💖 **Love**
    * 🔥 **Anger**
    * ⚡ **Fear**
    * 😲 **Surprise**
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware for external access or frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for static assets and templates
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they do not exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize Jinja2 Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse, summary="Serve Web Interface")
async def home(request: Request):
    """
    Renders the modern EmotiNet web application interface.
    """
    model_info = classifier.get_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "emotions": EMOTION_LABELS,
            "emotion_details": EMOTION_DETAILS,
            "benchmarks": MODEL_BENCHMARKS,
            "model_info": model_info
        }
    )


@app.get("/health", summary="API Health Check")
async def health_check():
    """
    Returns API operational health, model status, and runtime engine.
    """
    info = classifier.get_info()
    return {
        "status": "healthy",
        "model_loaded": info["is_model_loaded"],
        "engine": info["engine"],
        "version": info["version"],
        "supported_emotions": EMOTION_LABELS
    }


@app.get("/api/info", response_model=ModelInfoResponse, summary="Get Model Metadata & Benchmarks")
async def get_model_info():
    """
    Returns detailed architectural specs and benchmark comparisons across RNN, LSTM, GRU, and BiGRU.
    """
    return classifier.get_info()


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Predict Emotion for Single Text",
    status_code=status.HTTP_200_OK
)
async def predict_emotion(payload: PredictionRequest):
    """
    Analyzes input text and returns:
    - Dominant predicted emotion with confidence score
    - Full probability distribution across all 6 emotion classes
    - Emotional sentiment polarity (Positive, Negative, Neutral)
    - Qualitative intensity and contextual description
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input cannot be empty."
        )

    try:
        response = classifier.predict_single(payload.text)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post(
    "/api/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Predict Emotion for Batch of Texts",
    status_code=status.HTTP_200_OK
)
async def predict_emotion_batch(payload: BatchPredictionRequest):
    """
    Processes a list of text sentences in batch, returning individual predictions
    and aggregated emotion distribution statistics.
    """
    if not payload.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Texts list cannot be empty."
        )

    try:
        response = classifier.predict_batch(payload.texts)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )
