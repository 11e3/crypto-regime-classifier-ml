#!/usr/bin/env python
"""Debug ensemble predictions."""

from pathlib import Path

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models.deep import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier


def main():
    # Load data
    print("Loading data...")
    df = load_ohlcv("data/BTC.parquet")

    # Extract features
    extractor = FeatureExtractor(include_advanced=True)
    features = extractor.transform(df)

    # Generate labels
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Use last 20% as test set
    n_test = int(len(features) * 0.2)
    test_features = features.iloc[-n_test:]
    test_labels = labels.iloc[-n_test:]

    # Load models
    model_dir = Path("models")
    lstm = LSTMClassifier.load(model_dir / "regime_classifier_lstm_v2.pt")
    transformer = TransformerClassifier.load(model_dir / "regime_classifier_transformer_v2.pt")
    cnn_lstm = CNNLSTMClassifier.load(model_dir / "regime_classifier_cnn_lstm_v2.pt")

    print(f"LSTM n_classes: {lstm.n_classes}")
    print(f"Transformer n_classes: {transformer.n_classes}")
    print(f"CNN-LSTM n_classes: {cnn_lstm.n_classes}")

    # Get probabilities from each model
    print("\nProbabilities from first 5 samples:")

    lstm_proba = lstm.predict_proba(test_features)
    print(f"\nLSTM proba columns: {lstm_proba.columns.tolist()}")
    print(f"LSTM proba shape: {lstm_proba.shape}")
    print(lstm_proba.head())

    trans_proba = transformer.predict_proba(test_features)
    print(f"\nTransformer proba columns: {trans_proba.columns.tolist()}")
    print(f"Transformer proba shape: {trans_proba.shape}")
    print(trans_proba.head())

    cnn_proba = cnn_lstm.predict_proba(test_features)
    print(f"\nCNN-LSTM proba columns: {cnn_proba.columns.tolist()}")
    print(f"CNN-LSTM proba shape: {cnn_proba.shape}")
    print(cnn_proba.head())

    # Check if columns are in same order
    print("\n\nColumn check:")
    print(f"LSTM columns: {lstm_proba.columns.tolist()}")
    print(f"Transformer columns: {trans_proba.columns.tolist()}")
    print(f"CNN-LSTM columns: {cnn_proba.columns.tolist()}")

    # Manual average
    print("\n\nManual average (first 5 rows):")
    avg_proba = (lstm_proba + trans_proba + cnn_proba) / 3
    print(avg_proba.head())

    # Get predictions from average
    print("\nPredictions from averaged probabilities:")
    preds = avg_proba.idxmax(axis=1)
    print(preds.head(10))

    # Compare with labels
    common_idx = preds.index.intersection(test_labels.index)
    y_true = test_labels.loc[common_idx]
    y_pred = preds.loc[common_idx]

    accuracy = (y_true == y_pred).mean()
    print(f"\nManual ensemble accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
