import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import List
import talib
from utils.math_util import fit_parabola

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def detect_cup_handle_patterns(df: pd.DataFrame) -> List[dict]:
    df = df.copy()
    df["atr"] = calculate_atr(df)

    results = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    atr = df["atr"].values

    avg_candle_size = np.mean(np.abs(highs - lows))
    
    for i in range(300, len(df) - 60):
        for cup_len in range(30, 301):
            cup_start = i - cup_len - 50
            cup_end = i - 50
            handle_start = cup_end
            handle_end = i
            if cup_start < 0:
                continue

            try:
                # Fit parabola to cup
                cup_closes = closes[cup_start:cup_end]
                x = np.arange(len(cup_closes))
                coeffs, r2, y_fit = fit_parabola(x, cup_closes)

                if r2 < 0.85 or coeffs[0] <= 0:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "r2": float(r2),
                        "valid": False,
                        "invalid_reason": "V-shape / low R²"
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

                # Rim checks
                left_rim_close = closes[cup_start]
                right_rim_close = closes[cup_end - 1]
                left_rim_high = highs[cup_start]
                right_rim_high = highs[cup_end - 1]
                avg_rim = (left_rim_close + right_rim_close) / 2
                rim_price = max(left_rim_close, right_rim_close)
                max_rim_high = max(left_rim_high, right_rim_high)

                if abs(left_rim_close - right_rim_close) / avg_rim > 0.10:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Rim close mismatch > 10%"
                    })
                    continue

                # 🔴 Rim highs mismatch check
                rim_diff = abs(left_rim_high - right_rim_high) / np.mean([left_rim_high, right_rim_high])
                if rim_diff > 0.10:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Rim highs mismatch > 10%"
                    })
                    continue

                # Volume slope
                vol_slope, *_ = linregress(np.arange(len(volumes[cup_start:cup_end])), volumes[cup_start:cup_end])
                if vol_slope > 0:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Cup volume increasing"
                    })
                    continue

                # Handle metrics
                handle_closes = closes[handle_start:handle_end]
                handle_high = highs[handle_start:handle_end].max()
                handle_low = lows[handle_start:handle_end].min()

                if handle_high > max_rim_high:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "Handle high exceeds rim highs"
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

                retrace = (rim_price - handle_low) / depth if depth else 0
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

                # Breakout validation (ATR + volume)
                breakout_candle = df.iloc[i]
                atr_value = atr[i]
                breakout_strength = (breakout_candle["close"] - handle_high) > 1.5 * atr_value

                if not breakout_strength:
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "No strong price breakout (ATR rule failed)"
                    })
                    continue

                recent_vol = volumes[i - 14:i]
                if breakout_candle["volume"] < 1.5 * np.mean(recent_vol):
                    results.append({
                        "start_time": df.index[cup_start],
                        "end_time": df.index[i],
                        "valid": False,
                        "invalid_reason": "No breakout volume spike"
                    })
                    continue
                
                # ✅ Final valid pattern
                results.append({
                    "start_time": df.index[cup_start],
                    "end_time": df.index[i],
                    "cup_depth": float(depth),
                    "cup_duration": cup_end - cup_start,
                    "handle_duration": handle_duration,
                    "handle_high": float(handle_high),
                    "handle_low": float(handle_low),
                    "handle_retrace_ratio": float(retrace),
                    "r2": float(r2),
                    "breakout_time": df.index[i],
                    "breakout_volume": float(breakout_candle["volume"]),
                    "volume_slope": float(vol_slope),
                    "breakout_valid": breakout_strength,
                    "atr_value": float(atr_value),
                    "valid": True,
                    "invalid_reason": ""
                })
                print(f"Pattern evaluated: {df.index[cup_start]} → {df.index[i]}")
                if sum(p["valid"] for p in results) >= 2:
                    print("Multiple valid patterns found, stopping further checks.")
                    return results

            except Exception as e:
                results.append({
                    "start_time": df.index[cup_start],
                    "end_time": df.index[i],
                    "valid": False,
                    "invalid_reason": f"Exception: {str(e)}"
                })

            

    return results

def detect_cup_handle_patterns_loose(df: pd.DataFrame) -> list:
    results = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    avg_candle_size = np.mean(np.abs(highs - lows))
    atr = talib.ATR(highs, lows, closes, timeperiod=14)
    counter = 0

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
            print(f"{counter} ✅  Pattern detected from {df.index[cup_start]} to {df.index[i]}")
            counter += 1

    return results