import pandas as pd
import numpy as np
from scipy.stats import linregress

def extract_features(patterns, df):
    feature_rows = []

    for p in patterns:
        if not p.get("valid"):
            continue 

        try:
            cup_start = pd.to_datetime(p["start_time"])
            cup_end = cup_start + pd.Timedelta(minutes=p["cup_duration"])
            handle_start = cup_end
            handle_end = handle_start + pd.Timedelta(minutes=p["handle_duration"])
            breakout_time = pd.to_datetime(p["breakout_time"])

            cup_df = df.loc[cup_start:cup_end]
            handle_df = df.loc[handle_start:handle_end]

            handle_depth = p["handle_high"] - p["handle_low"]
            handle_retrace_ratio = handle_depth / p["cup_depth"] if p["cup_depth"] != 0 else 0

            breakout_price = df.loc[breakout_time]["close"]
            post_breakout_window = df.loc[breakout_time : breakout_time + pd.Timedelta(minutes=30)]
            if not post_breakout_window.empty:
                max_post_breakout = post_breakout_window["high"].max()
                breakout_strength_pct = (max_post_breakout - breakout_price) / breakout_price
            else:
                breakout_strength_pct = 0.0

            vol_series = cup_df["volume"]
            if len(vol_series) >= 2:
                x = np.arange(len(vol_series))
                slope, _, _, _, _ = linregress(x, vol_series)
                volume_slope = slope
            else:
                volume_slope = 0.0

            breakout_volume = df.loc[breakout_time]["volume"]

            feature_rows.append({
                "start_time": cup_start,
                "r2": p["r2"],
                "cup_depth": p["cup_depth"],
                "cup_duration": p["cup_duration"],
                "handle_duration": p["handle_duration"],
                "handle_retrace_ratio": handle_retrace_ratio,
                "breakout_strength_pct": breakout_strength_pct,
                "volume_slope": volume_slope,
                "breakout_volume": breakout_volume
            })

        except Exception as e:
            print(f"⚠️ Feature extraction failed for pattern starting at {p['start_time']}: {e}")
            continue

    return pd.DataFrame(feature_rows)
