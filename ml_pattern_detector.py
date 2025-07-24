import pandas as pd
from pattern_detector import detect_cup_handle_patterns
from ml_feature_extractor import extract_features
import joblib
import os

MODEL_PATH = "data/market-data/processed/model/pattern_rf_model.pkl"

def detect_patterns_with_ml(df, confidence_threshold=0.75):
    patterns = detect_cup_handle_patterns(df)

    if not patterns:
        return []

    features_df = extract_features(patterns, df)

    if features_df.empty:
        return []

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    feature_cols = [
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]

    X = features_df[feature_cols]
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    features_df["confidence"] = probabilities

    filtered_patterns = []
    for pattern, confidence in zip(patterns, probabilities):
        pattern["ml_confidence"] = round(float(confidence), 4)
        pattern["ml_valid"] = bool(confidence >= confidence_threshold)
        if pattern["ml_valid"]:
            filtered_patterns.append(pattern)

    return filtered_patterns
