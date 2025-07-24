import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

FEATURE_PATH = "data/market-data/processed/doc/pattern_features_for_labeling.csv"
MODEL_PATH = "data/market-data/processed/model/pattern_rf_model.pkl"

def train_model():
    df = pd.read_csv(FEATURE_PATH, parse_dates=["start_time"])

    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]

    if df.empty:
        print("No valid labeled data found.")
        return

    feature_cols = [
        "r2", "cup_depth", "cup_duration", "handle_duration",
        "handle_retrace_ratio", "breakout_strength_pct",
        "volume_slope", "breakout_volume"
    ]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    importances = clf.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"🔍 {col:<25}: {imp:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
