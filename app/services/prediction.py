from typing import Dict

from app.config import SUPPORTED_ASSETS, LOOKBACK_DAYS
from app.models.loader import model_registry
from app.services.data_loader import build_latest_feature_vector


def generate_signal(asset_name: str) -> Dict:
    """
    Generates an UP/DOWN/NEUTRAL signal and probabilities for a given asset.
    """
    if asset_name not in SUPPORTED_ASSETS:
        raise ValueError(f"Unsupported asset: {asset_name}")

    if asset_name not in model_registry.models:
        raise ValueError(f"No model loaded for asset: {asset_name}")

    ticker = SUPPORTED_ASSETS[asset_name]
    model = model_registry.models[asset_name]
    scaler = model_registry.scalers[asset_name]

    # Build latest feature vector (internally uses fixed FEATURE_COLS)
    latest_features = build_latest_feature_vector(
        ticker=ticker,
        lookback_days=LOOKBACK_DAYS
    )

    X_latest = latest_features.values  # shape (1, n_features)
    X_scaled = scaler.transform(X_latest)

    proba = model.predict_proba(X_scaled)[0]
    # Assuming proba[1] is probability of UP
    prob_up = float(proba[1])
    prob_down = float(proba[0])

    if prob_up > 0.55:
        signal = "UP"
    elif prob_down > 0.55:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    explanation = (
        f"Model indicates {signal} signal with "
        f"{prob_up:.2f} probability of upward move and "
        f"{prob_down:.2f} probability of downward move "
        f"in the next ~24 hours (research prototype, not financial advice)."
    )

    return {
        "asset": asset_name,
        "signal": signal,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "explanation": explanation
    }
