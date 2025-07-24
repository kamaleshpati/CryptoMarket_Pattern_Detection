import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import pandas as pd
import joblib
from detectors.pattern_detector import detect_cup_handle_patterns
from ml.ml_feature_extractor import extract_features

# Paths
MODEL_PATH = "data/market-data/model/pattern_sgd_model.pkl"
RAW_DATA_PATH = "data/market-data/raw/binance_1m.csv"

def test_model_loads():
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    model_bundle = joblib.load(MODEL_PATH)
    assert "model" in model_bundle and "scaler" in model_bundle, "Model bundle missing keys"
    print("✅ Model and scaler loaded successfully.")

def test_pattern_detection_not_empty():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    patterns = detect_cup_handle_patterns(df)
    assert len(patterns) > 0, "No patterns detected"
    print(f"✅ {len(patterns)} patterns detected successfully.")

def test_feature_columns_are_numeric():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    patterns = detect_cup_handle_patterns(df)
    features_df = extract_features(patterns, df)
    expected_cols = [
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature: {col}"
        assert pd.api.types.is_numeric_dtype(features_df[col]), f"Feature {col} is not numeric"
    print("✅ All feature columns are present and numeric.")

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
    print("✅ ML confidences are above zero and match the number of patterns.")

def test_prediction_probabilities_range():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    patterns = detect_cup_handle_patterns(df)
    features_df = extract_features(patterns, df)

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

    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities not in [0, 1] range"
    print("✅ All predicted probabilities fall within [0, 1].")

if __name__ == "__main__":
    print("\n📊 Running ML Pipeline Tests...\n")
    test_model_loads()
    test_pattern_detection_not_empty()
    test_feature_columns_are_numeric()
    test_model_predictions_above_zero()
    test_prediction_probabilities_range()
    print("\n🎉 All tests passed successfully!\n")
