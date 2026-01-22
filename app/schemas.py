from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text to analyze for emotional tone and sentiment",
        example="I just got my dream job offer today and I am thrilled!"
    )

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of text samples to analyze simultaneously"
    )

class EmotionScore(BaseModel):
    emotion: str
    label: str
    score: float
    percentage: float
    color: str
    emoji: str
    sentiment: str

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    confidence_percentage: float
    emoji: str
    secondary_emoji: str
    color: str
    sentiment: str
    intensity: str
    description: str
    probabilities: Dict[str, float]
    breakdown: List[EmotionScore]
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    total: int
    results: List[PredictionResponse]
    dominant_emotion: str
    emotion_distribution: Dict[str, int]
    processing_time_ms: float

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    architecture: str
    engine: str
    model_file: str
    tokenizer_file: str
    classes: List[str]
    max_sequence_length: int
    emotion_details: Dict[str, Any]
    benchmarks: List[Dict[str, Any]]
