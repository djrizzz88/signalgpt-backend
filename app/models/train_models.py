import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import SUPPORTED_ASSETS, PREDICTION_HORIZON_DAYS, MODELS_DIR
from app.services.data_loader import prepare_training_data


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_save_models():
    ensure_models_dir()

    for asset_name, ticker in SUPPORTED_ASSETS.items():
        print(f"\n==============================")
        print(f"Training model for {asset_name} ({ticker})")
        print(f"==============================")

        X, y, feature_cols = prepare_training_data(
            ticker=ticker,
            period="5y",
            interval="1d",
            horizon_days=PREDICTION_HORIZON_DAYS
        )

        print(f"Data shape for {asset_name}: X={X.shape}, y={y.shape}")

        # If no data, skip this asset
        if X.shape[0] == 0 or y.shape[0] == 0:
            print(f"WARNING: No usable data for {asset_name}, skipping model training.")
            continue

        # Time-ordered split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Simple RandomForest classifier
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_scaled, y_train)

        # Quick evaluation
        y_pred = clf.predict(X_test_scaled)
        print(f"\nClassification report for {asset_name}:")
        print(classification_report(y_test, y_pred))

        # Save model and scaler
        model_path = os.path.join(MODELS_DIR, f"{asset_name}_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"{asset_name}_scaler.pkl")

        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"Saved model to   {model_path}")
        print(f"Saved scaler to  {scaler_path}")


if __name__ == "__main__":
    train_and_save_models()
