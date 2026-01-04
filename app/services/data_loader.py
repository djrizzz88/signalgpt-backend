import datetime as dt

import pandas as pd
import yfinance as yf

from app.config import LOOKBACK_DAYS

# Fixed feature columns for the whole project
FEATURE_COLS = ["return_1d", "ma_5", "ma_10", "ma_20", "vol_5", "vol_10"]


def fetch_historical_data(ticker: str,
                          period: str = "5y",
                          interval: str = "1d") -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker using yfinance.
    Flatten MultiIndex columns (('Close','EURUSD=X') -> 'Close').
    """
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    # Drop missing rows
    data = data.dropna()

    # If yfinance gives MultiIndex columns, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    return data


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple technical features for modelling.
    """
    df = df.copy()

    # Daily returns
    df["return_1d"] = df["Close"].pct_change()

    # Moving averages
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # Volatility (std of returns)
    returns = df["Close"].pct_change()
    df["vol_5"] = returns.rolling(window=5).std()
    df["vol_10"] = returns.rolling(window=10).std()

    df = df.dropna()
    return df


def create_labels_for_direction(df: pd.DataFrame,
                                horizon_days: int = 1) -> pd.DataFrame:
    """
    Create binary label: 1 if future close (t + horizon) > current close, else 0.
    Uses NumPy arrays to avoid alignment issues.
    """
    df = df.copy()

    future_close = df["Close"].shift(-horizon_days)
    mask = future_close.notna()

    df = df.loc[mask].copy()
    future_close = future_close.loc[mask]
    current_close = df["Close"]

    target_up = (future_close.to_numpy() > current_close.to_numpy()).astype(int)

    df["future_close"] = future_close
    df["target_up"] = target_up

    return df


def prepare_training_data(ticker: str,
                          period: str = "5y",
                          interval: str = "1d",
                          horizon_days: int = 1):
    """
    Complete pipeline: fetch data, add features, create labels,
    and return X, y, feature_cols.
    """
    raw = fetch_historical_data(ticker, period=period, interval=interval)
    print("Raw columns:", raw.columns.tolist())

    feat = add_technical_features(raw)
    print("With features columns:", feat.columns.tolist())

    labelled = create_labels_for_direction(feat, horizon_days=horizon_days)
    print("Labelled columns:", labelled.columns.tolist())

    cols = list(FEATURE_COLS)
    print("Using feature cols:", cols)

    X = labelled.loc[:, cols].copy()
    y = labelled["target_up"].astype(int).copy()

    feature_cols = list(cols)
    return X, y, feature_cols


def build_latest_feature_vector(ticker: str,
                                lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    For live prediction: fetch recent data, compute features,
    and return the last row as a feature vector.
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days + 30)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True
    )

    df = df.dropna()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = add_technical_features(df)

    cols = list(FEATURE_COLS)
    latest = df.iloc[-1:].loc[:, cols].copy()
    return latest
