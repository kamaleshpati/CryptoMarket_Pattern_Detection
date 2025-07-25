````markdown
# 📊 Crypto Market Pattern Detection (BTCUSDT - Binance 1m)

A hybrid **Rule-Based + Machine Learning** system to detect **Cup & Handle** patterns in crypto market data. Uses Binance 1-minute candles and visualizes detected patterns via a Dash dashboard.

---

## 🚀 Getting Started

### 1️⃣ Download & Merge Binance Data

```bash
python download_and_merge.py
````

* Downloads 1-minute BTCUSDT data using Binance API
* Merges raw CSV files into:

```
data/market-data/raw/binance_1m.csv
```

---

### 2️⃣ Run the Detection + ML Pipeline

```bash
python main.py
```

Performs:

* ✅ Rule-based pattern detection
* ✅ Auto-labeling
* ✅ Feature extraction
* ✅ Model training (or fallback)
* ✅ ML confidence scoring
* ✅ PNG chart generation

Outputs:

* `report_rule.csv`: Rule-only patterns
* `report_ml.csv`: ML-enhanced patterns
* `.png` charts in `data/market-data/processed/pattern-charts/`

---

| Path                                                               | Description                                                                                                                           |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `data/market-data/processed/doc/report_rule.csv`                   | ✅ Contains **rule-based detected patterns**. Each row has start/end timestamps, depth, duration, r², and invalidation reason if any.  |
| `data/market-data/processed/doc/report_ml.csv`                     | ✅ Contains **ML-enhanced pattern analysis**. Same as above but includes `ml_confidence` and `ml_valid` fields.                        |
| `data/market-data/processed/doc/pattern_features_for_labeling.csv` | 🧠 Extracted features for each detected pattern, used for ML training. Also includes auto-generated label (0 or 1).                   |
| `data/market-data/model/pattern_sgd_model.pkl`                     | 🤖 Trained ML model bundle, including the `SGDClassifier` and its `StandardScaler`. Loaded or updated each time you run the pipeline. |
| `data/market-data/processed/pattern-charts/cup_handle_*.png`       | 📉 PNG charts of **rule-based valid patterns** (named `cup_handle_1.png`, `cup_handle_2.png`, etc.).                                  |
| `data/market-data/processed/pattern-charts/ml_cup_handle_*.png`    | 📈 PNG charts of **ML-validated patterns** only, with high confidence. (named `ml_cup_handle_1.png`, etc.)                            |


### 3️⃣ Launch Interactive Dashboard

```bash
python app.py
```

* Browse price + patterns by day
* 🟥 Red overlays: Rule-based patterns
* 🟩 Green overlays: ML-validated patterns

---

## 🧠 Machine Learning Details

* Model: `SGDClassifier` (log-loss)
* Features extracted from rule-based pattern geometry
* Trained incrementally on auto-labeled patterns

#### Manually Retrain Model

```bash
python ml/train_incremental.py
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

Covers:

* Pattern detection output
* Feature extraction validity
* ML predictions non-zero and in \[0, 1]

---

## 📁 Project Structure

```
crypto-market-pattern/
├── data/
│   └── market-data/
│       ├── raw/                # Binance 1m candles
│       ├── processed/doc/      # Pattern reports
│       └── model/              # Trained model
├── detectors/                  # Rule-based pattern logic
├── ml/                         # Feature extraction & model training
├── config/                     # config.json and loader
├── tests/                      # ML pipeline integration tests
├── main.py                     # Full detection + ML runner
├── app.py                      # Dash dashboard
├── download_and_merge.py       # Data downloader
├── README.md
```

---

## ⚙️ Config Management

All file paths and thresholds live in:

```
data/configuration/config.json
```

Use them anywhere via:

```python
from config.config_loader import RAW_DATA_PATH, MODEL_PATH, ...
```

---

## ✅ Requirements

* Python 3.9+
* pandas, numpy
* scikit-learn, joblib
* dash, plotly
* ta-lib
* pytest

```
Built By KAMALESH PATI
```
