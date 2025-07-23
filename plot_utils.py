# plot_utils.py

import os
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit

def plot_and_save_pattern(df, pattern, save_path):
    """
    Plots the Cup and Handle pattern using Plotly and saves as PNG using Kaleido.
    """
    start = pattern["start_time"]
    end = pattern["end_time"]
    df_pattern = df.loc[start:end].copy()
    df_pattern.reset_index(inplace=True)

    # 1. Create figure
    fig = go.Figure()

    # 2. Price line
    fig.add_trace(go.Scatter(
        x=df_pattern['timestamp'],
        y=df_pattern['close'],
        mode='lines',
        name='Price',
        line=dict(color='gray')
    ))

    # 3. Fit cup parabola and overlay
    cup_len = pattern["cup_duration"]
    cup_df = df_pattern.iloc[:cup_len]
    x = np.arange(len(cup_df))
    y = cup_df['close'].values

    def parabola(x, a, b, c):
        return a * x**2 + b * x + c

    popt, _ = curve_fit(parabola, x, y)
    y_fit = parabola(x, *popt)

    fig.add_trace(go.Scatter(
        x=cup_df['timestamp'],
        y=y_fit,
        mode='lines',
        name='Fitted Cup',
        line=dict(color='purple', width=2, dash='dash')
    ))

    # 4. Handle region (blue highlight)
    handle_start = cup_len
    handle_end = cup_len + pattern["handle_duration"]
    handle_df = df_pattern.iloc[handle_start:handle_end]

    fig.add_trace(go.Scatter(
        x=handle_df['timestamp'],
        y=handle_df['close'],
        mode='lines',
        name='Handle',
        line=dict(color='blue', width=2)
    ))

    # 5. Breakout candle
    breakout_time = pattern["breakout_time"]
    breakout_price = df.loc[breakout_time]['close']

    fig.add_trace(go.Scatter(
        x=[breakout_time],
        y=[breakout_price],
        mode='markers+text',
        name='Breakout',
        marker=dict(color='green', size=10, symbol='star'),
        text=["Breakout"],
        textposition="top center"
    ))

    # 6. Layout
    fig.update_layout(
        title=f"Cup & Handle Pattern: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        showlegend=True,
        width=1000,
        height=500
    )

    # 7. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path, engine="kaleido")

    print(f"✅ Saved pattern image to {save_path}")
