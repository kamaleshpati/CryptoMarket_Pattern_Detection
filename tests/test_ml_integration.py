import pandas as pd
from detectors.pattern_detector import detect_cup_handle_patterns
from ml.ml_feature_extractor import extract_features
import joblib
import os

MODEL_PATH = "data/market-data/model/pattern_sgd_model.pkl"
RAW_DATA_PATH = "data/market-data/raw/binance_1m.csv"

def test_model_predictions_above_zero():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    patterns = detect_cup_handle_patterns(df)
    features_df = extract_features(patterns, df)

    assert not features_df.empty, "Feature extraction returned empty DataFrame"

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    X = features_df[[
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]]
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]

    assert (proba > 0).any(), "All ML confidences are zero or less"
    assert len(proba) == len(patterns), "Mismatch between patterns and probabilities"
    print("All tests passed: ML confidences are above zero and match the number of patterns.")