import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Load data once
df = pd.read_csv("data/market-data/raw/binance_1m.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

patterns_df = pd.read_csv(
    "data/market-data/processed/doc/report.csv",
    parse_dates=["start_time", "end_time"],
    low_memory=False
)

app = dash.Dash(__name__)
app.title = "Crypto Binance Cup & Handle Visualizer"

# Layout
app.layout = html.Div([
    html.H2("BTC Pattern Visualizer"),

    dcc.DatePickerSingle(
        id='date-picker',
        date=str(df.index.date.min()),
        min_date_allowed=df.index.date.min(),
        max_date_allowed=df.index.date.max(),
        display_format='YYYY-MM-DD'
    ),

    html.Button("Previous Day", id="prev-day", n_clicks=0),
    html.Button("Next Day", id="next-day", n_clicks=0),

    dcc.Graph(id='chart', config={"displayModeBar": True}),
])

# Callbacks
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

    # Clamp within range
    min_date = df.index.date.min()
    max_date = df.index.date.max()
    date = max(min_date, min(date.date(), max_date))
    return str(date)

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

    # Overlay patterns for this day
    day_patterns = patterns_df[
        (patterns_df["start_time"] >= start) & (patterns_df["end_time"] < end) & (patterns_df["valid"] == True)
    ]

    for _, row in day_patterns.iterrows():
        fig.add_vrect(
            x0=row["start_time"], x1=row["end_time"],
            fillcolor="red", opacity=0.25, layer="below", line_width=0,
            annotation_text="Cup & Handle", annotation_position="top left"
        )

    fig.update_layout(
        title=f"Price Chart for {date.date()}",
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
        template="plotly_white"
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
