import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectors import detect_cup_handle_patterns

import pandas as pd
from ml import extract_features
import joblib

from config import RAW_DATA_PATH, MODEL_PATH

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