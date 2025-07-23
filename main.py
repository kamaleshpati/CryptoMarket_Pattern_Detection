# main.py

import os
import pandas as pd
from pattern_detector import detect_cup_handle_patterns
from plot_utils import plot_and_save_pattern
from dashboard_generator import generate_pattern_dashboard

# Paths
RAW_DATA_PATH = "data/market-data/raw/binance_1m.csv"
OUTPUT_DIR = "data/market-data/processed/media"
REPORT_PATH = "data/market-data/processed/doc/report.csv"
DASHBOARD_PATH = "data/market-data/processed/web/pattern_dashboard.html"

def main():
    # Step 1: Load 1-minute OHLCV data
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Step 2: Detect patterns
    patterns = detect_cup_handle_patterns(df)
    print(f"\n✅ Found {len([p for p in patterns if p['valid']])} valid Cup and Handle patterns:\n")

    # Step 3: Plot and save valid patterns
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, pattern in enumerate(patterns):
        if pattern["valid"]:
            print(f"{i+1}. From {pattern['start_time']} to {pattern['end_time']} | "
                  f"Depth: {pattern['cup_depth']:.2f} | R²: {pattern['r2']:.2f} | "
                  f"Breakout: {pattern['breakout_time']}")
            
            filename = f"cup_handle_{i+1}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            plot_and_save_pattern(df, pattern, save_path)

    # Step 4: Save CSV report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    pd.DataFrame(patterns).to_csv(REPORT_PATH, index=False)
    print(f"\n📄 Report saved to {REPORT_PATH}")

    # # Step 5: Generate interactive dashboard
    # generate_pattern_dashboard(
    #     data_path=RAW_DATA_PATH,
    #     patterns_path=REPORT_PATH,
    #     output_path=DASHBOARD_PATH
    # )

    # print(f"\n🌐 Open the interactive dashboard here:")
    # print(f"file://{os.path.abspath(DASHBOARD_PATH)}")

if __name__ == "__main__":
    main()
