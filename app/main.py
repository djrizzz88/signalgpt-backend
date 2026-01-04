from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.config import SUPPORTED_ASSETS
from app.models.loader import model_registry
from app.services.prediction import generate_signal



class PredictRequest(BaseModel):
    asset: str
    horizon_hours: Optional[int] = 24  # reserved for future use


class PredictResponse(BaseModel):
    asset: str
    signal: str
    prob_up: float
    prob_down: float
    explanation: str


app = FastAPI(
    title="SignalGPT Backend",
    description="Lightweight financial signal prediction chatbot (research prototype).",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models_on_startup():
    # Load all models when the server starts
    model_registry.load_all()


@app.get("/health")
def health_check():
    return {"status": "ok", "supported_assets": list(SUPPORTED_ASSETS.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    asset = request.asset.upper()

    if asset not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported asset: {asset}")

    try:
        result = generate_signal(asset_name=asset)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
