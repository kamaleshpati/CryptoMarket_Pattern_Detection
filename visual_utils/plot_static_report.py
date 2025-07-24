import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from detectors.pattern_detector import detect_cup_handle_patterns

def load_binance_data(path="data/market-data/raw/binance_1m.csv") -> pd.DataFrame:
    """
    Loads Binance 1-minute OHLCV data and sets 'timestamp' as index.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1000
pio.kaleido.scope.default_height = 600

def plot_cup_handle_pattern(df, pattern, output_path):
    start = pattern["start_time"]
    end = pattern["end_time"]

    pattern_df = df.loc[start:end]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pattern_df.index, y=pattern_df['close'], mode='lines', name='Close'))

    # Add cup arc if available
    if "r2" in pattern and pattern["r2"] > 0.85:
        cup_start = start
        cup_end = cup_start + pd.Timedelta(minutes=pattern["cup_duration"])
        cup_df = df.loc[cup_start:cup_end]
        x = list(range(len(cup_df)))
        y = cup_df["close"].values
        coeffs = np.polyfit(x, y, 2)
        y_fit = np.polyval(coeffs, x)
        fig.add_trace(go.Scatter(x=cup_df.index, y=y_fit, mode='lines', name='Fitted Cup Arc', line=dict(color='orange', dash='dot')))

    # Mark handle and breakout
    handle_start = cup_end
    handle_end = handle_start + pd.Timedelta(minutes=pattern["handle_duration"])
    fig.add_vrect(x0=handle_start, x1=handle_end, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Handle")
    fig.add_vline(x=pattern["breakout_time"], line=dict(color="green", dash="dash"), annotation_text="Breakout")

    fig.update_layout(title=f"Cup and Handle Pattern | R²: {pattern['r2']:.2f}", xaxis_title="Time", yaxis_title="Price", template="plotly_white")
    fig.write_image(output_path)

def main():
    data_path = "data/market-data/raw/binance_1m.csv"
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    print("🔍 Detecting patterns...")
    patterns = detect_cup_handle_patterns(df)

    valid_patterns = [p for p in patterns if p["valid"]]
    print(f"✅ Detected {len(valid_patterns)} valid Cup & Handle patterns.")

    output_dir = "data/market-data/processed/media"
    os.makedirs(output_dir, exist_ok=True)

    for idx, pattern in enumerate(valid_patterns):
        filename = f"cup_handle_{idx+1}.png"
        output_path = os.path.join(output_dir, filename)
        plot_cup_handle_pattern(df, pattern, output_path)
        print(f"📈 Saved: {output_path}")

    # Save report
    pd.DataFrame(patterns).to_csv("data/market-data/processed/doc/report_rule.csv", index=False)
    print("📄 Detection report saved.")

if __name__ == "__main__":
    main()
