import os
import pandas as pd
import joblib

from detectors.pattern_detector import detect_cup_handle_patterns_loose
from ml.ml_feature_extractor import extract_features
from ml.train_model import train_incremental  
from visual_utils.plot_utils import plot_and_save_pattern
# from app import run_server 

RAW_DATA_PATH = "data/market-data/raw/binance_1m.csv"
OUTPUT_DIR = "data/market-data/processed/media"
FEATURE_PATH = "data/market-data/processed/doc/pattern_features_for_labeling.csv"
REPORT_RULE_PATH = "data/market-data/processed/doc/report_rule.csv"
REPORT_ML_PATH = "data/market-data/processed/doc/report_ml.csv"
MODEL_PATH = "data/market-data/model/pattern_sgd_model.pkl"
CONFIDENCE_THRESHOLD = 0.5
MIN_VALID_PATTERNS = 30

def auto_label(row, df):
    try:
        cup_start = row["start_time"]
        cup_end = cup_start + pd.Timedelta(minutes=row["cup_duration"])
        cup_volume = df.loc[cup_start:cup_end]["volume"]
        avg_cup_vol = cup_volume.mean() or 1
        return int(
            row["r2"] >= 0.90 and
            row["breakout_strength_pct"] > 0.015 and
            row["handle_retrace_ratio"] < 0.35 and
            row["volume_slope"] >= 0 and
            row["breakout_volume"] > 1.25 * avg_cup_vol
        )
    except Exception:
        return 0

def main():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Step 1: Rule-Based Pattern Detection
    patterns = detect_cup_handle_patterns_loose(df)
    valid_patterns = [p for p in patterns if p["valid"]]
    print(f"\n✅ Rule-based: {len(valid_patterns)} valid patterns detected")

    pretrained_used = False

    if len(valid_patterns) < MIN_VALID_PATTERNS:
        print(f"⚠️ Only {len(valid_patterns)} valid patterns found (<{MIN_VALID_PATTERNS})")
        if os.path.exists(MODEL_PATH):
            print("🤖 Using pretrained model for ML scoring...")
            pretrained_used = True
        else:
            print("🛑 No pretrained model available. Exiting.")
            return

    # Step 2: Feature Extraction
    features_df = extract_features(patterns, df)
    if features_df.empty:
        print("❌ Feature extraction returned empty. Exiting.")
        return

    # Step 3: Auto-label
    features_df["label"] = features_df.apply(lambda row: auto_label(row, df), axis=1)
    os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)
    features_df.to_csv(FEATURE_PATH, index=False)
    print(f"🧠 Features saved to {FEATURE_PATH}")

    # Step 4: Train model if not exists
    if not os.path.exists(MODEL_PATH):
        print("⚙️ No model found. Training initial model...")
        train_incremental()
    else:
        print("📦 Existing model found." + (" (pretrained fallback)" if pretrained_used else ""))

    # Step 5: Load Model and Apply to All Patterns
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    feature_cols = [
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]

    try:
        X = features_df[feature_cols]
        X_scaled = scaler.transform(X)
        y_proba = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        print(f"❌ Error in ML inference: {e}")
        return

    # Step 6: Add ML Confidence & Validity
    for pattern, confidence in zip(patterns, y_proba):
        pattern["ml_confidence"] = round(float(confidence), 4)
        pattern["ml_valid"] = bool(confidence >= CONFIDENCE_THRESHOLD)

    # Step 7: Save Rule-Based Report
    os.makedirs(os.path.dirname(REPORT_RULE_PATH), exist_ok=True)
    pd.DataFrame(patterns).to_csv(REPORT_RULE_PATH, index=False)
    print(f"📄 Rule-based report saved: {REPORT_RULE_PATH}")

    # Step 8: Save ML-Enhanced Report
    os.makedirs(os.path.dirname(REPORT_ML_PATH), exist_ok=True)
    pd.DataFrame(patterns).to_csv(REPORT_ML_PATH, index=False)
    print(f"📄 ML-enhanced report saved: {REPORT_ML_PATH}")

    # Step 9: Plot ML-Valid and Valid Patterns Only
    ml_patterns = [p for p in patterns if p.get("ml_valid")]
    print(f"📈 {len(ml_patterns)} ML-valid patterns found (confidence >= {CONFIDENCE_THRESHOLD})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, pattern in enumerate(ml_patterns):
        required_keys = ["cup_duration", "handle_duration", "start_time", "end_time"]
        if not all(k in pattern for k in required_keys):
            print(f"⚠️ Skipping ML pattern {i+1}: missing required keys")
            continue

        filename = f"ml_cup_handle_{i+1}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plot_and_save_pattern(df, pattern, save_path)

    print(f"📸 Saved {len(ml_patterns)} ML-validated pattern plots.")

    for i, pattern in enumerate(patterns):
        if pattern["valid"]:
            required_keys = ["cup_duration", "handle_duration", "start_time", "end_time"]
            if not all(k in pattern for k in required_keys):
                print(f"⚠️ Skipping pattern {i+1}: missing required keys")
                continue

            print(f"{i+1}. From {pattern['start_time']} to {pattern['end_time']} | "
                  f"Depth: {pattern['cup_depth']:.2f} | R²: {pattern['r2']:.2f} | "
                  f"Breakout: {pattern['breakout_time']}")

            filename = f"cup_handle_{i+1}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            plot_and_save_pattern(df, pattern, save_path)
        else:
            pass

    print(f"📸 Saved {len(patterns)} Lib-validated pattern plots.")

    # Step 10: Retrain model if not using fallback
    if not pretrained_used:
        print("📚 Retraining model incrementally...")
        train_incremental()
    else:
        print("🚫 Skipping model training (using pretrained fallback).")

    # Step 11: Launch Dashboard (do it separately , as it might  start without processing too)
    # run_server()

if __name__ == "__main__":
    main()
