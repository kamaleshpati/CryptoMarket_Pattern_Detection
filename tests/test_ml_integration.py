import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectors import detect_cup_handle_patterns_loose

import pandas as pd
from ml import extract_features
import joblib

from config import RAW_DATA_PATH, MODEL_PATH

def test_model_predictions_on_first_5_days():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Limit to first 5 days of data
    start_time = df.index.min()
    end_time = start_time + pd.Timedelta(days=5)
    df = df[(df.index >= start_time) & (df.index < end_time)]
    print(f"📅 Using data from {start_time} to {end_time} — {len(df)} rows")

    patterns = detect_cup_handle_patterns_loose(df)
    assert len(patterns) > 0, "No patterns detected in first 5 days"

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

    print(f"✅ Predictions: min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"📊 Probas > 0: {(proba > 0).sum()} / {len(proba)}")

    assert (proba > 0).any(), "All ML confidences are zero or less"
    assert len(proba) == len(features_df), "Mismatch between features and predicted probabilities"

    print("✅ All tests passed: ML confidences are above zero and match pattern count.")

