import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
import joblib
from ml import extract_features
from detectors import detect_cup_handle_patterns_loose
from config import RAW_DATA_PATH, MODEL_PATH

def load_df_first_n_days(n=5):
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()
    start_time = df.index.min()
    end_time = start_time + pd.Timedelta(days=n)
    df = df[(df.index >= start_time) & (df.index < end_time)]
    print(f"📅 Loaded data from {start_time} to {end_time} — {len(df)} rows")
    return df

def test_model_loads():
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    model_bundle = joblib.load(MODEL_PATH)
    assert "model" in model_bundle and "scaler" in model_bundle, "Model bundle missing keys"
    print("✅ Model and scaler loaded successfully.")

def test_pattern_detection_not_empty():
    df = load_df_first_n_days()
    patterns = detect_cup_handle_patterns_loose(df)
    assert len(patterns) > 0, "No patterns detected in small dataset"
    print(f"✅ Detected {len(patterns)} pattern(s).")

def test_feature_columns_are_numeric():
    df = load_df_first_n_days()
    patterns = detect_cup_handle_patterns_loose(df)
    features_df = extract_features(patterns, df)
    
    if features_df.empty:
        print("⚠️ No features extracted. Skipping numeric column checks.")
        return

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
    df = load_df_first_n_days()
    patterns = detect_cup_handle_patterns_loose(df)
    features_df = extract_features(patterns, df)

    if features_df.empty:
        print("⚠️ No features extracted. Skipping prediction check.")
        return

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

    assert len(proba) == len(features_df), "Mismatch between feature rows and probabilities"
    assert (proba > 0).any(), "All ML confidences are zero or less"
    print(f"✅ ML confidences valid. Probas > 0: {(proba > 0).sum()} / {len(proba)}")

def test_prediction_probabilities_range():
    df = load_df_first_n_days()
    patterns = detect_cup_handle_patterns_loose(df)
    features_df = extract_features(patterns, df)

    if features_df.empty:
        print("⚠️ No features extracted. Skipping probability range check.")
        return

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
    print(f"✅ All {len(proba)} predicted probabilities fall within [0, 1].")

if __name__ == "__main__":
    print("\n📊 Running ML Pipeline Tests on Small Subset...\n")
    test_model_loads()
    test_pattern_detection_not_empty()
    test_feature_columns_are_numeric()
    test_model_predictions_above_zero()
    test_prediction_probabilities_range()
    print("\n🎉 All tests passed (or skipped safely if dataset was too small)!\n")
