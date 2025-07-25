import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

from detectors import detect_cup_handle_patterns
from .ml_feature_extractor import extract_features
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from config import MODEL_PATH, RAW_DATA_PATH

def auto_label(row, df):
    try:
        cup_start = row["start_time"]
        cup_end = cup_start + timedelta(minutes=row["cup_duration"])
        cup_volume = df.loc[cup_start:cup_end]["volume"]
        avg_cup_vol = cup_volume.mean() or 1
        return int(
            row["r2"] >= 0.90 and
            row["breakout_strength_pct"] > 0.015 and
            row["handle_retrace_ratio"] < 0.35 and
            row["volume_slope"] >= 0 and
            row["breakout_volume"] > 1.25 * avg_cup_vol
        )
    except:
        return 0

# --- Streaming Trainer ---
def update_model_live(df):
    patterns = detect_cup_handle_patterns(df)
    if not patterns:
        print("No new patterns found.")
        return

    features_df = extract_features(patterns, df)
    if features_df.empty:
        print("No features extracted.")
        return

    features_df["label"] = features_df.apply(lambda row: auto_label(row, df), axis=1)
    features_df = features_df[features_df["label"].isin([0, 1])]

    if features_df.empty:
        print("No auto-labeled data to train on.")
        return

    feature_cols = [
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]
    X = features_df[feature_cols]
    y = features_df["label"]

    # --- Load or initialize model ---
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        scaler = bundle["scaler"]
        X_scaled = scaler.transform(X)
        model.partial_fit(X_scaled, y)
        print("🔁 Updated existing model with new patterns.")
    else:
        model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        model.partial_fit(X_scaled, y, classes=np.array([0, 1]))
        print("🆕 Trained new incremental model.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"💾 Model updated and saved to: {MODEL_PATH}")

if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    update_model_live(df.tail(1000))  # Example: only train on last N rows
