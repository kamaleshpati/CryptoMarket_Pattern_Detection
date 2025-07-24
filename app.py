import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import os

# === Paths ===
DATA_PATH = "data/market-data/raw/binance_1m.csv"
RULE_REPORT_PATH = "data/market-data/processed/doc/report_rule.csv"
ML_REPORT_PATH = "data/market-data/processed/doc/report_ml.csv"

# === Load Main Price Data ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Raw data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === Safely Load Rule-Based Patterns ===
patterns_rules_df = pd.DataFrame()
if os.path.exists(RULE_REPORT_PATH) and os.path.getsize(RULE_REPORT_PATH) > 0:
    try:
        patterns_rules_df = pd.read_csv(
            RULE_REPORT_PATH, parse_dates=["start_time", "end_time"], low_memory=False
        )
    except pd.errors.EmptyDataError:
        print(f"⚠️ Warning: {RULE_REPORT_PATH} exists but is empty.")

# === Safely Load ML-Based Patterns ===
patterns_ml_df = pd.DataFrame()
if os.path.exists(ML_REPORT_PATH) and os.path.getsize(ML_REPORT_PATH) > 0:
    try:
        patterns_ml_df = pd.read_csv(
            ML_REPORT_PATH, parse_dates=["start_time", "end_time"], low_memory=False
        )
    except pd.errors.EmptyDataError:
        print(f"⚠️ Warning: {ML_REPORT_PATH} exists but is empty.")

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Crypto Binance Cup & Handle Visualizer"

app.layout = html.Div([
    html.H2("BTC Cup & Handle Pattern Visualizer"),
    
    dcc.DatePickerSingle(
        id='date-picker',
        date=str(df.index.date.min()),
        min_date_allowed=df.index.date.min(),
        max_date_allowed=df.index.date.max(),
        display_format='YYYY-MM-DD'
    ),
    
    html.Div([
        html.Button("⬅️ Previous Day", id="prev-day", n_clicks=0),
        html.Button("Next Day ➡️", id="next-day", n_clicks=0),
    ], style={"margin": "10px 0"}),

    dcc.Graph(id='chart', config={"displayModeBar": True}),
])

# === Date Navigation Callback ===
@app.callback(
    Output("date-picker", "date"),
    Input("prev-day", "n_clicks"),
    Input("next-day", "n_clicks"),
    Input("date-picker", "date"),
)
def update_date(prev_clicks, next_clicks, selected_date):
    ctx = dash.callback_context
    if not ctx.triggered:
        return selected_date

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    date = datetime.strptime(selected_date, "%Y-%m-%d")

    if button_id == "prev-day":
        date -= timedelta(days=1)
    elif button_id == "next-day":
        date += timedelta(days=1)

    min_date = df.index.date.min()
    max_date = df.index.date.max()
    date = max(min_date, min(date.date(), max_date))

    return str(date)

# === Chart Update Callback ===
@app.callback(
    Output("chart", "figure"),
    Input("date-picker", "date")
)
def update_chart(date):
    date = pd.to_datetime(date)
    start = date
    end = date + timedelta(days=1)

    df_day = df[(df.index >= start) & (df.index < end)]

    fig = go.Figure(data=[
        go.Candlestick(
            x=df_day.index,
            open=df_day["open"],
            high=df_day["high"],
            low=df_day["low"],
            close=df_day["close"],
            name="Price"
        )
    ])

    # Rule-based overlays (red)
    if not patterns_rules_df.empty:
        day_rules = patterns_rules_df[
            (patterns_rules_df["start_time"] >= start) &
            (patterns_rules_df["end_time"] < end) &
            (patterns_rules_df["valid"] == True)
        ]
        for _, row in day_rules.iterrows():
            fig.add_vrect(
                x0=row["start_time"], x1=row["end_time"],
                fillcolor="red", opacity=0.25, line_width=0,
                annotation_text="Rule-based", annotation_position="top left"
            )

    # ML-based overlays (green)
    if not patterns_ml_df.empty:
        day_ml = patterns_ml_df[
            (patterns_ml_df["start_time"] >= start) &
            (patterns_ml_df["end_time"] < end) &
            (patterns_ml_df.get("ml_valid", True) == True)
        ]
        for _, row in day_ml.iterrows():
            fig.add_vrect(
                x0=row["start_time"], x1=row["end_time"],
                fillcolor="green", opacity=0.25, line_width=0,
                annotation_text="ML-based", annotation_position="top right"
            )

    fig.update_layout(
        title=f"Price Chart with Pattern Overlays – {date.date()}",
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
        template="plotly_white"
    )

    return fig

def run_server():
    app.run(debug=True)

if __name__ == "__main__":
    run_server()
