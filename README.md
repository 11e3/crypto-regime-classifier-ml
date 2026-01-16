# Crypto Regime Classifier ML

**Machine learning model for market regime classification in the Crypto Quant Ecosystem.**

Part of: [crypto-quant-system](https://github.com/11e3/crypto-quant-system) → [bt](https://github.com/11e3/bt) → [crypto-bot](https://github.com/11e3/crypto-bot) → **[crypto-regime-classifier-ml](https://github.com/11e3/crypto-regime-classifier-ml)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![GCS](https://img.shields.io/badge/Storage-GCS-blue.svg)](https://cloud.google.com/storage)

## Ecosystem Role

```
┌─────────────────────────────────────────────────────────────────┐
│                    Crypto Quant Ecosystem                       │
├─────────────────────────────────────────────────────────────────┤
│  crypto-quant-system     │  Dashboard & data pipeline          │
│    └── Model viewer      │  - Displays regime predictions      │
├──────────────────────────┼──────────────────────────────────────┤
│  bt                      │  Backtesting engine                 │
│    └── Regime backtest   │  - Tests regime-aware strategies    │
├──────────────────────────┼──────────────────────────────────────┤
│  crypto-bot              │  Live trading bot                   │
│    └── Model consumer    │  - Loads .pkl from GCS ◄────────┐   │
│                          │  - Adjusts strategy by regime   │   │
├──────────────────────────┼─────────────────────────────────┤   │
│  crypto-regime-ml        │  Market regime classifier       │   │
│  (this repo)             │                                 │   │
│    ├── Feature eng       │  - Extracts market features     │   │
│    ├── Model training    │  - Trains classifier            │   │
│    └── GCS upload        │  - Uploads .pkl to GCS ─────────┘   │
└──────────────────────────┴──────────────────────────────────────┘
```

## Regime Types

| Regime | Description | Strategy Adjustment |
|--------|-------------|---------------------|
| `BULL_TREND` | Strong uptrend, high momentum | Full position, trend follow |
| `BEAR_TREND` | Strong downtrend | Reduce exposure, defensive |
| `SIDEWAYS` | Range-bound, low volatility | Mean reversion, tight stops |
| `HIGH_VOL` | High volatility, uncertain | Reduce size, wider stops |

## Features Used

### Price Features
- Returns (1d, 5d, 20d)
- Volatility (rolling std)
- RSI, MACD
- Bollinger Band position

### Volume Features
- Volume ratio (vs MA)
- OBV trend

### Market Structure
- Higher highs/lows count
- MA alignment (5, 20, 60)
- ATR percentile

## Quick Start

### Installation

```bash
git clone <repository-url>
cd crypto-regime-classifier-ml

pip install -r requirements.txt
```

### Train Model

```bash
# Train with default settings
python train.py --data ../crypto-quant-system/data/

# Train with custom parameters
python train.py \
  --data ../crypto-quant-system/data/ \
  --model random_forest \
  --features all \
  --output models/
```

### Upload to GCS

```bash
# Upload trained model
python upload.py --model models/regime_classifier_v1.pkl

# Or use CLI
gsutil cp models/regime_classifier_v1.pkl gs://your-bucket/models/
```

## Project Structure

```
crypto-regime-classifier-ml/
├── train.py                # Training entry point
├── upload.py               # GCS upload script
├── src/
│   ├── features/
│   │   ├── price.py        # Price-based features
│   │   ├── volume.py       # Volume-based features
│   │   └── structure.py    # Market structure features
│   ├── models/
│   │   ├── classifier.py   # Regime classifier
│   │   └── ensemble.py     # Ensemble methods
│   ├── labeling/
│   │   └── regime.py       # Regime labeling logic
│   └── utils/
│       ├── data.py         # Data loading
│       └── gcs.py          # GCS utilities
├── notebooks/
│   ├── eda.ipynb           # Exploratory analysis
│   └── evaluation.ipynb    # Model evaluation
└── models/
    └── .gitkeep
```

## Training Pipeline

```python
from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.models import RegimeClassifier

# 1. Load data
data = load_ohlcv("data/BTC.csv")

# 2. Label regimes (for training)
labeler = RegimeLabeler(
    trend_threshold=0.02,
    vol_percentile=80
)
labels = labeler.label(data)

# 3. Extract features
extractor = FeatureExtractor()
features = extractor.transform(data)

# 4. Train model
model = RegimeClassifier(model_type="random_forest")
model.fit(features, labels)

# 5. Save and upload
model.save("models/regime_classifier_v1.pkl")
upload_to_gcs("models/regime_classifier_v1.pkl")
```

## Model Interface

```python
# Used by crypto-bot
class RegimeClassifier:
    def predict(self, features: pd.DataFrame) -> str:
        """Returns: 'BULL_TREND', 'BEAR_TREND', 'SIDEWAYS', 'HIGH_VOL'"""
        
    def predict_proba(self, features: pd.DataFrame) -> dict:
        """Returns probability for each regime"""
        
    def get_feature_names(self) -> list:
        """Returns required feature names for prediction"""
```

## GCS Integration

### Upload Model

```python
from google.cloud import storage

def upload_model(local_path: str, version: str):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"models/regime_classifier_{version}.pkl")
    blob.upload_from_filename(local_path)
    
    # Also update 'latest' pointer
    latest = bucket.blob("models/regime_classifier_latest.pkl")
    latest.upload_from_filename(local_path)
```

### Version Management

```
gs://your-bucket/models/
├── regime_classifier_v1.pkl
├── regime_classifier_v2.pkl
├── regime_classifier_latest.pkl  → symlink to current
└── metadata/
    └── v2.json  # training params, metrics
```

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | > 60% | Overall classification accuracy |
| F1 (macro) | > 0.55 | Balanced performance across regimes |
| Regime persistence | > 3 days | Avoid frequent regime switches |
| Backtest improvement | > 10% | Sharpe improvement vs baseline |

## Integration with crypto-bot

```python
# In crypto-bot/bot/regime.py
from google.cloud import storage
import pickle

class RegimeManager:
    def __init__(self):
        self.model = self._load_model()
        
    def _load_model(self):
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("models/regime_classifier_latest.pkl")
        
        with blob.open("rb") as f:
            return pickle.load(f)
    
    def get_current_regime(self, ohlcv: pd.DataFrame) -> str:
        features = self._extract_features(ohlcv)
        return self.model.predict(features)
    
    def adjust_position_size(self, base_size: float, regime: str) -> float:
        adjustments = {
            "BULL_TREND": 1.0,
            "BEAR_TREND": 0.5,
            "SIDEWAYS": 0.7,
            "HIGH_VOL": 0.3,
        }
        return base_size * adjustments[regime]
```

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check .

# Notebook experiments
jupyter lab notebooks/
```

## Roadmap

- [ ] HMM-based regime detection
- [ ] Deep learning features (LSTM embeddings)
- [ ] Multi-timeframe regime fusion
- [ ] Online learning / model update
- [ ] A/B testing framework

## License

MIT License

---

**Version**: 0.1.0 (Planned) | **Ecosystem**: Crypto Quant System | **Output**: GCS .pkl
