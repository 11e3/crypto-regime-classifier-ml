#!/usr/bin/env python
"""Debug label distribution."""

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv


def main():
    # Load data
    print("Loading data...")
    df = load_ohlcv("data/BTC.parquet")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Generate labels
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Use last 20% as test set
    n_test = int(len(labels) * 0.2)
    train_labels = labels.iloc[:-n_test]
    test_labels = labels.iloc[-n_test:]

    print("\n=== Full Dataset ===")
    print(labels.value_counts())
    print(labels.value_counts(normalize=True))

    print("\n=== Training Set (80%) ===")
    print(train_labels.value_counts())
    print(train_labels.value_counts(normalize=True))

    print("\n=== Test Set (20%) ===")
    print(test_labels.value_counts())
    print(test_labels.value_counts(normalize=True))

    # Check test set date range
    print(f"\nTest set date range: {test_labels.index.min()} to {test_labels.index.max()}")

    # Check if test period is consistently NOT_BULL
    print("\n=== Test Set Label Changes ===")
    changes = test_labels != test_labels.shift(1)
    change_points = test_labels[changes]
    print(change_points.head(20))


if __name__ == "__main__":
    main()
