# pattern_detector.py

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import talib

def fit_parabola(x: np.ndarray, y: np.ndarray):
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    popt, _ = curve_fit(parabola, x, y)
    y_fit = parabola(x, *popt)
    r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)
    return popt, r2, y_fit

def detect_cup_handle_patterns(df: pd.DataFrame) -> list:
    results = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    avg_candle_size = np.mean(np.abs(highs - lows))
    atr = talib.ATR(highs, lows, closes, timeperiod=14)

    for i in range(300, len(df) - 60):
        for cup_len in range(30, 301):
            cup_start = i - cup_len - 50
            cup_end = i - 50
            handle_start = cup_end
            handle_end = i
            if cup_start < 0:
                continue

            try:
                cup_closes = closes[cup_start:cup_end]
                x = np.arange(len(cup_closes))
                popt, r2, y_fit = fit_parabola(x, cup_closes)
                if r2 < 0.85 or popt[0] <= 0:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "r2": float(r2),
                        "valid": False,
                        "invalid_reason": "Cup not U-shaped or low R²"
                    })
                    continue

                depth = np.max(y_fit) - np.min(y_fit)
                if depth < 2 * avg_candle_size:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "cup_depth": float(depth),
                        "valid": False,
                        "invalid_reason": "Cup too shallow"
                    })
                    continue

                left_rim = cup_closes[0]
                right_rim = cup_closes[-1]
                avg_rim = (left_rim + right_rim) / 2
                if abs(left_rim - right_rim) / avg_rim > 0.10:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Rim mismatch > 10%"
                    })
                    continue

                # Volume trend check
                vol_slope, *_ = linregress(np.arange(len(volumes[cup_start:cup_end])), volumes[cup_start:cup_end])
                if vol_slope > 0:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Cup volume increasing"
                    })
                    continue

                handle_closes = closes[handle_start:handle_end]
                handle_high = np.max(handle_closes)
                handle_low = np.min(handle_closes)

                if handle_high > max(left_rim, right_rim):
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Handle high above rim"
                    })
                    continue

                if handle_low < np.min(cup_closes):
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Handle breaks below cup"
                    })
                    continue

                retrace = (handle_high - handle_low) / depth
                handle_duration = handle_end - handle_start
                if retrace > 0.4:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Handle retracement > 40%"
                    })
                    continue

                if handle_duration < 5 or handle_duration > 50:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Handle duration invalid"
                    })
                    continue

                breakout_candle = df.iloc[i]
                if breakout_candle['close'] <= handle_high + 1.5 * atr[i]:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "No strong price breakout"
                    })
                    continue

                recent_vol = volumes[i-14:i]
                if breakout_candle['volume'] < 1.5 * np.mean(recent_vol):
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "No breakout volume spike"
                    })
                    continue

                # If all checks pass: VALID
                results.append({
                    "start_time": df.index[cup_start],
                    "end_time": df.index[i],
                    "cup_depth": float(depth),
                    "cup_duration": cup_end - cup_start,
                    "handle_duration": handle_duration,
                    "handle_high": float(handle_high),
                    "handle_low": float(handle_low),
                    "r2": float(r2),
                    "breakout_time": df.index[i],
                    "breakout_volume": float(breakout_candle['volume']),
                    "volume_slope": float(vol_slope),
                    "valid": True,
                    "invalid_reason": ""
                })

                if sum(p["valid"] for p in results) >= 30:
                    return results

            except Exception as e:
                results.append({
                    "start_time": df.index[cup_start],
                    "end_time": df.index[i],
                    "valid": False,
                    "invalid_reason": f"Exception: {str(e)}"
                })
            print(f"Pattern detected from {df.index[cup_start]} to {df.index[i]}")

    return results
