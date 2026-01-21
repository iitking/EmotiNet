import os
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "BiGRU_Modle.keras"))
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", str(BASE_DIR / "tokenizer.pkl"))

# Preprocessing parameters (matching trained model specs)
MAX_SEQUENCE_LENGTH = 50
MAX_WORDS = 10000
PADDING = "post"
TRUNCATING = "post"

# Emotion Classes Mapping (Index 0-5 from dair-ai/emotion dataset)
EMOTION_LABELS = [
    "sadness",   # 0
    "joy",       # 1
    "love",      # 2
    "anger",     # 3
    "fear",      # 4
    "surprise"   # 5
]

# Rich Emotion Metadata for Frontend and Insights
EMOTION_DETAILS = {
    "joy": {
        "label": "Joy",
        "emoji": "😊",
        "secondary_emoji": "✨",
        "color": "#F59E0B",      # Amber / Gold
        "bg_color": "rgba(245, 158, 11, 0.15)",
        "border_color": "rgba(245, 158, 11, 0.4)",
        "sentiment": "Positive",
        "intensity": "High Energy",
        "description": "Expresses happiness, delight, celebration, excitement, and overall optimism.",
        "gradient": "from-amber-500 to-yellow-400"
    },
    "sadness": {
        "label": "Sadness",
        "emoji": "😢",
        "secondary_emoji": "🌧️",
        "color": "#3B82F6",      # Bright Blue
        "bg_color": "rgba(59, 130, 246, 0.15)",
        "border_color": "rgba(59, 130, 246, 0.4)",
        "sentiment": "Negative",
        "intensity": "Low Energy",
        "description": "Reflects sorrow, heartbreak, loss, loneliness, grief, and melancholy.",
        "gradient": "from-blue-600 to-indigo-500"
    },
    "love": {
        "label": "Love",
        "emoji": "💖",
        "secondary_emoji": "🥰",
        "color": "#EC4899",      # Rose / Pink
        "bg_color": "rgba(236, 72, 153, 0.15)",
        "border_color": "rgba(236, 72, 153, 0.4)",
        "sentiment": "Positive",
        "intensity": "Warm & Deep",
        "description": "Indicates affection, romantic fondness, deep empathy, gratitude, and care.",
        "gradient": "from-pink-500 to-rose-400"
    },
    "anger": {
        "label": "Anger",
        "emoji": "🔥",
        "secondary_emoji": "😡",
        "color": "#EF4444",      # Red
        "bg_color": "rgba(239, 68, 68, 0.15)",
        "border_color": "rgba(239, 68, 68, 0.4)",
        "sentiment": "Negative",
        "intensity": "High Energy",
        "description": "Depicts frustration, fury, resentment, irritation, and hostility.",
        "gradient": "from-red-600 to-orange-500"
    },
    "fear": {
        "label": "Fear",
        "emoji": "⚡",
        "secondary_emoji": "😨",
        "color": "#8B5CF6",      # Violet / Purple
        "bg_color": "rgba(139, 92, 246, 0.15)",
        "border_color": "rgba(139, 92, 246, 0.4)",
        "sentiment": "Negative",
        "intensity": "Tense & Anxious",
        "description": "Signifies anxiety, dread, panic, phobia, terror, and impending distress.",
        "gradient": "from-purple-600 to-violet-500"
    },
    "surprise": {
        "label": "Surprise",
        "emoji": "😲",
        "secondary_emoji": "🎉",
        "color": "#06B6D4",      # Cyan
        "bg_color": "rgba(6, 182, 212, 0.15)",
        "border_color": "rgba(6, 182, 212, 0.4)",
        "sentiment": "Neutral",
        "intensity": "Sudden Spike",
        "description": "Denotes unexpected astonishment, shock, awe, marvel, and amazement.",
        "gradient": "from-cyan-500 to-teal-400"
    }
}

# Model Architecture benchmark metadata from notebook experiments
MODEL_BENCHMARKS = [
    {
        "model": "BiGRU (Bidirectional GRU)",
        "test_loss": 0.334,
        "test_accuracy": 0.887,
        "parameters": "3.5M",
        "status": "Production Selected",
        "highlight": True
    },
    {
        "model": "Standard GRU",
        "test_loss": 0.412,
        "test_accuracy": 0.854,
        "parameters": "1.8M",
        "status": "Baseline",
        "highlight": False
    },
    {
        "model": "LSTM",
        "test_loss": 0.448,
        "test_accuracy": 0.842,
        "parameters": "2.1M",
        "status": "Baseline",
        "highlight": False
    },
    {
        "model": "Simple RNN",
        "test_loss": 0.689,
        "test_accuracy": 0.761,
        "parameters": "1.4M",
        "status": "Baseline",
        "highlight": False
    }
]

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
