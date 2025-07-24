import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_PATH = "data/market-data/processed/doc/pattern_features_for_labeling.csv"
MODEL_PATH = "data/market-data/model/pattern_sgd_model.pkl"

def train_incremental():
    print("📂 Loading labeled feature data...")
    df = pd.read_csv(FEATURE_PATH, parse_dates=["start_time"])
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]

    if df.empty:
        print("❌ No labeled data found. Exiting.")
        return

    feature_cols = [
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]
    X = df[feature_cols]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print("🔎 Class balance:")
    print(f"🧪 Train: {y_train.value_counts().to_dict()}")
    print(f"🧪 Test : {y_test.value_counts().to_dict()}")

    # Load or initialize model and scaler
    if os.path.exists(MODEL_PATH):
        print("📦 Loading existing model...")
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        scaler = bundle["scaler"]
    else:
        print("🆕 Creating new incremental model...")
        model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
        print(f"✅ Model initialized and saved to {MODEL_PATH}")
        return

    # Continue training
    X_train = scaler.transform(X_train)
    model.partial_fit(X_train, y_train)

    # Evaluate
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if len(np.unique(y_test)) > 1:
        print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    else:
        print("⚠️ ROC-AUC cannot be computed — only one class in y_test.")

    # Save updated model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"\n💾 Updated incremental model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_incremental()
