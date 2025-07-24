import os
import pandas as pd
from pattern_detector import detect_cup_handle_patterns
from plot_utils import plot_and_save_pattern
from ml_feature_extractor import extract_features
# from dashboard_generator import generate_pattern_dashboard
# from ml_pattern_detector import detect_patterns_with_ml

RAW_DATA_PATH = "data/market-data/raw/binance_1m.csv"
OUTPUT_DIR = "data/market-data/processed/media"
REPORT_PATH = "data/market-data/processed/doc/report.csv"
FEATURE_PATH = "data/market-data/processed/doc/pattern_features_for_labeling.csv"
DASHBOARD_PATH = "data/market-data/processed/web/pattern_dashboard.html"

def main():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    patterns = detect_cup_handle_patterns(df)
    # patterns = detect_patterns_with_ml(df, confidence_threshold=0.75)
    print(f"\n✅ Found {len([p for p in patterns if p['valid']])} valid Cup and Handle patterns:\n")

    features_df = extract_features(patterns, df)

    def auto_label(row):
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
        except:
            return 0

    features_df["label"] = features_df.apply(auto_label, axis=1)

    os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)
    features_df.to_csv(FEATURE_PATH, index=False)
    print(f"\n🧠 Feature file auto-labeled and saved to: {FEATURE_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, pattern in enumerate(patterns):
        if pattern["valid"]:
            print(f"{i+1}. From {pattern['start_time']} to {pattern['end_time']} | "
                  f"Depth: {pattern['cup_depth']:.2f} | R²: {pattern['r2']:.2f} | "
                  f"Breakout: {pattern['breakout_time']}")
            
            filename = f"cup_handle_{i+1}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            plot_and_save_pattern(df, pattern, save_path)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    pd.DataFrame(patterns).to_csv(REPORT_PATH, index=False)
    print(f"\n📄 Report saved to {REPORT_PATH}")

    # Optional Dashboard (commented out)
    # generate_pattern_dashboard(
    #     data_path=RAW_DATA_PATH,
    #     patterns_path=REPORT_PATH,
    #     output_path=DASHBOARD_PATH
    # )
    # print(f"\n Open the interactive dashboard here:")
    # print(f"file://{os.path.abspath(DASHBOARD_PATH)}")

if __name__ == "__main__":
    main()
