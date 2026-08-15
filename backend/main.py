import os
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Social Media Crisis Detection API",
    description="Backend API for crisis detection using DistilBERT",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "distilbert_crisis_model"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AI SOCIAL MEDIA CRISIS DETECTION API")
print("=" * 60)

print("Model directory:")
print(MODEL_DIR)

print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    print("✅ DistilBERT model loaded successfully.")

except Exception as e:

    tokenizer = None
    model = None

    print("❌ Model loading failed.")
    print(e)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    text: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Social Media Crisis Detection API",
        "status": "running",
        "model": "DistilBERT"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None or tokenizer is None:

        return {
            "error": "Model is not loaded."
        }


    text = request.text.strip()


    if not text:

        return {
            "error": "Text cannot be empty."
        }


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )


    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if predicted_class == 1:

        label = "INFORMATIVE"

    else:

        label = "NOT_INFORMATIVE"


    return {

        "prediction": label,

        "confidence": round(
            confidence,
            4
        ),

        "confidence_percent": round(
            confidence * 100,
            2
        ),

        "text": text,

        "model": "DistilBERT"

    }


# ============================================================
# END
# ============================================================