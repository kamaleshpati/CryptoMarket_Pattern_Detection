# dashboard_generator.py

import pandas as pd
import plotly.graph_objects as go

def generate_pattern_dashboard(data_path, patterns_path, output_path):
    # Load OHLCV data
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Downsample to 5-minute OHLCV
    df_resampled = df.resample("5T").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    # Load pattern summary
    patterns = pd.read_csv(patterns_path, parse_dates=["start_time", "end_time"])

    # Start plot
    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df_resampled.index,
        open=df_resampled['open'],
        high=df_resampled['high'],
        low=df_resampled['low'],
        close=df_resampled['close'],
        name="Price"
    ))

    # Overlay valid pattern ranges
    for i, row in patterns.iterrows():
        if not row['valid']:
            continue

        fig.add_vrect(
            x0=row['start_time'], x1=row['end_time'],
            fillcolor="green", opacity=0.25,
            layer="below", line_width=0,
            annotation_text=f"Pattern {i+1}", annotation_position="top left"
        )

    # Layout
    fig.update_layout(
        title="Cup and Handle Patterns (5-Min Chart)",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=800
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"✅ Interactive dashboard saved to: {output_path}")
