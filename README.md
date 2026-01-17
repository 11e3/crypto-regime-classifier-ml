# Crypto Regime Classifier ML

**Machine learning model for market regime classification in the Crypto Quant Ecosystem.**

Part of: [crypto-quant-system](https://github.com/11e3/crypto-quant-system) → [bt](https://github.com/11e3/bt) → [crypto-bot](https://github.com/11e3/crypto-bot) → **[crypto-regime-classifier-ml](https://github.com/11e3/crypto-regime-classifier-ml)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-red.svg)](https://pytorch.org/)

## Research Results Summary

### Model Performance Comparison (Walk-Forward Validation)

| Model | Features | Accuracy | F1 Score | Notes |
|-------|----------|----------|----------|-------|
| **XGBoost (Ultra-5)** | 5 | **88.47% ± 12.87%** | **0.7839 ± 0.30** | Best efficiency |
| XGBoost (Full) | 51 | 88.83% ± 12.78% | 0.7505 ± 0.31 | Baseline |
| XGBoost (Reduced) | 18 | 87.47% ± 13.91% | 0.7901 ± 0.28 | Good balance |
| Random Forest | 51 | 87.34% ± 13.81% | 0.7263 ± 0.32 | - |
| LSTM | 51 | 66.00% ± 15.00% | 0.5500 ± 0.25 | Poor |

**Key Finding**: 5 features achieve nearly identical performance to 51 features.

### Selected Features (Ultra-5)

| Feature | Description | Importance |
|---------|-------------|------------|
| `return_20d` | 20-day momentum | Trend direction |
| `volatility` | 20-day rolling std | Risk level |
| `rsi` | 14-day RSI | Overbought/oversold |
| `ma_alignment` | MA(5,20,60) alignment | Trend strength |
| `volume_ratio_20` | Volume vs 20-day avg | Confirmation |

### Validation Methodology

- **Walk-Forward Validation**: 37 folds
- **Train Size**: 500 days
- **Test Size**: 60 days
- **Step Size**: 60 days (no overlap)
- **Data**: BTC daily OHLCV (2017-2024)

## Regime Types

| Regime | Description | Trading Implication |
|--------|-------------|---------------------|
| `BULL_TREND` | Strong uptrend, aligned MAs | Full position, trend follow |
| `NOT_BULL` | Downtrend or sideways | Reduce exposure, defensive |

> Note: Binary classification (2-class) outperformed 4-class in testing.

## Quick Start

### Installation

```bash
git clone https://github.com/11e3/crypto-regime-classifier-ml.git
cd crypto-regime-classifier-ml
pip install -r requirements.txt
```

### Train Model

```bash
# Train XGBoost with full features
python train.py --data data/BTC.parquet --model xgboost

# Train with LSTM
python train.py --data data/BTC.parquet --model lstm --epochs 50
```

### Export for Production

```bash
# Export Ultra-5 model (recommended)
python scripts/export/export_ultra5.py

# Output:
#   - models/regime_classifier_xgb_ultra5.joblib
#   - models/regime_feature_calculator.py
```

## Usage in crypto-quant-system

```python
import joblib
from regime_feature_calculator import predict_regime, calculate_regime_features

# Load model
clf = joblib.load("models/regime_classifier_xgb_ultra5.joblib")

# Option 1: Direct prediction
regime = predict_regime(clf, ohlcv_df)
# Returns: "BULL_TREND" or "NOT_BULL"

# Option 2: Get probabilities
features = calculate_regime_features(ohlcv_df)
X_scaled = clf["scaler"].transform(features[clf["feature_names"]])
proba = clf["model"].predict_proba(X_scaled)
# proba[:, 0] = BULL_TREND probability
# proba[:, 1] = NOT_BULL probability
```

### Required OHLCV Format

```python
# DataFrame with columns: open, high, low, close, volume
# Index: DatetimeIndex
# Minimum: 60 rows (for MA60 calculation)
```

## Project Structure

```
crypto-regime-classifier-ml/
├── train.py                    # Main training entry point
├── upload.py                   # GCS upload script
├── src/
│   ├── features/
│   │   ├── extractor.py        # Feature extraction pipeline
│   │   ├── price.py            # Price-based features
│   │   ├── volume.py           # Volume-based features
│   │   ├── structure.py        # Market structure features
│   │   └── advanced.py         # Advanced features (optional)
│   ├── models/
│   │   ├── classifier.py       # RegimeClassifier (RF, XGBoost)
│   │   ├── ensemble.py         # Model ensemble
│   │   ├── hybrid_ensemble.py  # Supervised + unsupervised
│   │   ├── deep/               # Deep learning models
│   │   │   ├── lstm.py         # LSTM classifier
│   │   │   ├── transformer.py  # Transformer classifier
│   │   │   └── cnn_lstm.py     # CNN-LSTM hybrid
│   │   └── unsupervised/       # Unsupervised models
│   │       ├── kmeans.py       # K-Means clustering
│   │       ├── gmm.py          # Gaussian Mixture Model
│   │       └── hmm.py          # Hidden Markov Model
│   ├── labeling/
│   │   └── regime.py           # Regime labeling logic
│   ├── validation/
│   │   ├── walk_forward.py     # Walk-forward validation
│   │   └── expanding_window.py # Expanding window validation
│   └── utils/
│       ├── data.py             # Data loading utilities
│       └── gcs.py              # GCS upload utilities
├── scripts/
│   ├── analysis/               # Analysis scripts
│   │   ├── analyze_pca.py      # PCA feature analysis
│   │   ├── test_minimal_features.py
│   │   └── walk_forward_validation.py
│   ├── export/                 # Export scripts
│   │   ├── export_ultra5.py    # 5-feature model export
│   │   └── export_for_quant_system.py
│   ├── training/               # Training scripts
│   │   └── train_*_ensemble.py
│   └── debug/                  # Debug utilities
├── models/                     # Trained model files
├── outputs/                    # Analysis outputs
└── data/                       # Training data
```

## Feature Engineering Details

### PCA Analysis Results

- **80% variance**: 10 components
- **90% variance**: 15 components
- **95% variance**: 22 components
- **22 highly correlated pairs** (>0.9 correlation) identified and removed

### Feature Categories

| Category | Full (51) | Reduced (18) | Ultra (5) |
|----------|-----------|--------------|-----------|
| Returns | 7 | 3 | 1 |
| Volatility | 6 | 2 | 1 |
| Momentum | 8 | 3 | 1 |
| Volume | 6 | 3 | 1 |
| Trend | 10 | 4 | 1 |
| Bollinger | 5 | 1 | 0 |
| Range | 9 | 2 | 0 |

## Model Export Format

```python
# Exported .joblib structure (no custom class dependencies)
{
    "model": XGBClassifier,           # Trained model
    "scaler": StandardScaler,         # Feature scaler
    "label_encoder": LabelEncoder,    # Label encoder
    "feature_names": ["return_20d", ...],  # Required features
    "classes": ["BULL_TREND", "NOT_BULL"], # Class names
    "performance": {
        "accuracy_mean": 0.8847,
        "accuracy_std": 0.1287,
        "f1_mean": 0.7839,
        "f1_std": 0.30,
    },
}
```

## Deep Learning Models

### Available Architectures

| Model | Description | Status |
|-------|-------------|--------|
| LSTM | Bidirectional LSTM with attention | Implemented |
| Transformer | Multi-head self-attention | Implemented |
| CNN-LSTM | Conv1D + LSTM hybrid | Implemented |

### Training Deep Models

```bash
# LSTM
python train.py --model lstm --seq-length 60 --epochs 100

# Transformer
python train.py --model transformer --seq-length 60 --epochs 100
```

> Note: Deep learning models showed worse performance than XGBoost in walk-forward validation. Likely due to limited data and regime shift patterns.

## GCS Integration

```bash
# Upload model
python upload.py --model models/regime_classifier_xgb_ultra5.joblib

# Or direct
gsutil cp models/regime_classifier_xgb_ultra5.joblib gs://your-bucket/models/
```

## Completed Research

- [x] Feature engineering (51 features)
- [x] PCA analysis and feature reduction (51 → 5)
- [x] Random Forest vs XGBoost comparison
- [x] LSTM, Transformer, CNN-LSTM implementation
- [x] Walk-forward validation (37 folds)
- [x] Supervised + Unsupervised ensemble
- [x] Model export for production

## Key Insights

1. **Simpler is better**: 5 features match 51 features performance
2. **XGBoost dominates**: Outperforms RF, LSTM, Transformer
3. **Deep learning disappoints**: LSTM 66% vs XGBoost 88%
4. **Binary classification**: 2-class better than 4-class for trading
5. **Walk-forward essential**: Time-series CV prevents data leakage

## Future Work

- [ ] Online learning / incremental updates
- [ ] Multi-asset regime correlation
- [ ] Regime transition prediction
- [ ] Confidence-based position sizing

## License

MIT License

---

**Version**: 1.0.0 | **Best Model**: XGBoost Ultra-5 (88.47% accuracy)
