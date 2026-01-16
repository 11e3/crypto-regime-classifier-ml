#!/usr/bin/env python
"""Debug ensemble predictions - check individual model accuracy manually."""

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models.deep import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier


def main():
    # Load data
    print("Loading data...")
    df = load_ohlcv("data/BTC.parquet")

    # Extract features - use same settings as training (no advanced features)
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {len(features.columns)}")

    # Generate labels
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Use last 20% as test set (same as training)
    n_test = int(len(features) * 0.2)
    test_features = features.iloc[-n_test:]
    test_labels = labels.iloc[-n_test:]

    print(f"\nTest samples: {len(test_features)}")
    print(f"Test labels distribution:")
    print(test_labels.value_counts())

    # Load models
    model_dir = Path("models")
    lstm = LSTMClassifier.load(model_dir / "regime_classifier_lstm_v2.pt")
    transformer = TransformerClassifier.load(model_dir / "regime_classifier_transformer_v2.pt")
    cnn_lstm = CNNLSTMClassifier.load(model_dir / "regime_classifier_cnn_lstm_v2.pt")

    print(f"\nLSTM expected features: {len(lstm.feature_names)}")
    print(f"Transformer expected features: {len(transformer.feature_names)}")
    print(f"CNN-LSTM expected features: {len(cnn_lstm.feature_names)}")

    # Check if feature names match
    print(f"\nFeature names match LSTM: {set(features.columns) == set(lstm.feature_names)}")
    missing_lstm = set(lstm.feature_names) - set(features.columns)
    extra_lstm = set(features.columns) - set(lstm.feature_names)
    if missing_lstm:
        print(f"Missing features for LSTM: {missing_lstm}")
    if extra_lstm:
        print(f"Extra features not in LSTM: {extra_lstm}")

    # Predict with each model
    print("\n=== LSTM ===")
    lstm_preds = lstm.predict(test_features)
    common_idx = lstm_preds.index.intersection(test_labels.index)
    lstm_acc = accuracy_score(test_labels.loc[common_idx], lstm_preds.loc[common_idx])
    print(f"Accuracy: {lstm_acc:.4f}")
    print(f"Prediction distribution:")
    print(lstm_preds.value_counts())

    print("\n=== Transformer ===")
    trans_preds = transformer.predict(test_features)
    common_idx = trans_preds.index.intersection(test_labels.index)
    trans_acc = accuracy_score(test_labels.loc[common_idx], trans_preds.loc[common_idx])
    print(f"Accuracy: {trans_acc:.4f}")
    print(f"Prediction distribution:")
    print(trans_preds.value_counts())

    print("\n=== CNN-LSTM ===")
    cnn_preds = cnn_lstm.predict(test_features)
    common_idx = cnn_preds.index.intersection(test_labels.index)
    cnn_acc = accuracy_score(test_labels.loc[common_idx], cnn_preds.loc[common_idx])
    print(f"Accuracy: {cnn_acc:.4f}")
    print(f"Prediction distribution:")
    print(cnn_preds.value_counts())

    # Manual ensemble with hard voting
    print("\n=== Manual Ensemble (Hard Voting) ===")
    common_idx = lstm_preds.index.intersection(trans_preds.index).intersection(cnn_preds.index).intersection(test_labels.index)

    import pandas as pd
    votes = pd.DataFrame({
        'lstm': lstm_preds.loc[common_idx],
        'transformer': trans_preds.loc[common_idx],
        'cnn_lstm': cnn_preds.loc[common_idx],
    })
    ensemble_preds = votes.mode(axis=1)[0]
    ensemble_acc = accuracy_score(test_labels.loc[common_idx], ensemble_preds)
    print(f"Accuracy: {ensemble_acc:.4f}")
    print(f"Prediction distribution:")
    print(ensemble_preds.value_counts())

    # Soft voting
    print("\n=== Manual Ensemble (Soft Voting) ===")
    lstm_proba = lstm.predict_proba(test_features)
    trans_proba = transformer.predict_proba(test_features)
    cnn_proba = cnn_lstm.predict_proba(test_features)

    avg_proba = (lstm_proba + trans_proba + cnn_proba) / 3
    soft_preds = avg_proba.idxmax(axis=1)
    common_idx = soft_preds.index.intersection(test_labels.index)
    soft_acc = accuracy_score(test_labels.loc[common_idx], soft_preds.loc[common_idx])
    print(f"Accuracy: {soft_acc:.4f}")
    print(f"Prediction distribution:")
    print(soft_preds.value_counts())

    print("\n=== Classification Report (Soft Ensemble) ===")
    print(classification_report(test_labels.loc[common_idx], soft_preds.loc[common_idx]))


if __name__ == "__main__":
    main()
