import os
from typing import Dict, Any

import joblib

from app.config import SUPPORTED_ASSETS, MODELS_DIR


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    def load_all(self):
        loaded_assets = []
        for asset_name in SUPPORTED_ASSETS.keys():
            model_path = os.path.join(MODELS_DIR, f"{asset_name}_model.pkl")
            scaler_path = os.path.join(MODELS_DIR, f"{asset_name}_scaler.pkl")

            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                print(f"Warning: model or scaler missing for {asset_name}, skipping.")
                continue

            self.models[asset_name] = joblib.load(model_path)
            self.scalers[asset_name] = joblib.load(scaler_path)
            loaded_assets.append(asset_name)

        print(f"Loaded models for assets: {loaded_assets}")


model_registry = ModelRegistry()
