#!/usr/bin/env python
"""Training entry point for regime classifier."""

import argparse
from pathlib import Path

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.models import (
    RegimeClassifier,
    EnsembleClassifier,
    LSTMClassifier,
    TransformerClassifier,
    CNNLSTMClassifier,
)
from src.utils.data import load_ohlcv, load_multiple_symbols, prepare_training_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a market regime classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data directory or CSV file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols to load (e.g., BTC ETH)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=[
            "random_forest", "gradient_boosting", "logistic", "ensemble",
            "lstm", "transformer", "cnn_lstm",
        ],
        help="Model type to train",
    )
    parser.add_argument(
        "--ensemble-models",
        type=str,
        nargs="+",
        default=["random_forest", "gradient_boosting"],
        help="Models to use in ensemble",
    )

    # Deep learning arguments
    parser.add_argument(
        "--seq-length",
        type=int,
        default=60,
        help="Sequence length for deep learning models",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden dimension size for deep learning models",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of layers for deep learning models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for deep learning training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs for deep learning",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for deep learning",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=3,
        choices=[3, 4],
        help="Number of classes (3: BULL/BEAR/SIDEWAYS, 4: includes HIGH_VOL)",
    )

    # Feature arguments
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        choices=["all", "price", "volume", "structure"],
        help="Feature types to use",
    )
    parser.add_argument(
        "--advanced-features",
        action="store_true",
        help="Include advanced features (recommended for deep learning)",
    )

    # Labeling arguments
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.02,
        help="Return threshold for trend classification",
    )
    parser.add_argument(
        "--vol-percentile",
        type=float,
        default=80,
        help="Volatility percentile for HIGH_VOL regime",
    )
    parser.add_argument(
        "--use-lookahead",
        action="store_true",
        help="Use lookahead labeling (for training only)",
    )

    # Training arguments
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="models/",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Model version string",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("Crypto Regime Classifier - Training")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    data_path = Path(args.data)

    if data_path.is_file():
        # Single file
        df = load_ohlcv(data_path)
        print(f"Loaded {len(df)} rows from {data_path}")
    else:
        # Directory with multiple files
        data_dict = load_multiple_symbols(data_path, symbols=args.symbols)
        if not data_dict:
            raise ValueError(f"No data files found in {data_path}")

        # Combine all data (for simplicity, using first symbol)
        # In production, you might want to train on combined data
        symbol = list(data_dict.keys())[0]
        df = data_dict[symbol]
        print(f"Using {symbol} data: {len(df)} rows")

    # Initialize feature extractor
    print("\n[2/5] Configuring features...")
    feature_config = {
        "include_price": args.features in ["all", "price"],
        "include_volume": args.features in ["all", "volume"],
        "include_structure": args.features in ["all", "structure"],
        "include_advanced": args.advanced_features,
    }
    extractor = FeatureExtractor(**feature_config)
    print(f"Feature types: {args.features}")
    print(f"Advanced features: {args.advanced_features}")
    print(f"Total features: {len(extractor.get_feature_names())}")

    # Initialize labeler
    print("\n[3/5] Configuring labeler...")
    labeler = RegimeLabeler(
        trend_threshold=args.trend_threshold,
        vol_percentile=args.vol_percentile,
        n_classes=args.n_classes,
    )
    print(f"Trend threshold: {args.trend_threshold}")
    print(f"Volatility percentile: {args.vol_percentile}")
    print(f"Number of classes: {args.n_classes}")

    # Prepare training data
    print("\n[4/5] Preparing training data...")
    X_train, X_test, y_train, y_test = prepare_training_data(
        df=df,
        feature_extractor=extractor,
        labeler=labeler,
        test_size=args.test_size,
        use_lookahead=args.use_lookahead,
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Show label distribution
    print("\nLabel distribution (training):")
    label_counts = y_train.value_counts()
    for label, count in label_counts.items():
        pct = count / len(y_train) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Check if using deep learning model
    is_deep_learning = args.model in ["lstm", "transformer", "cnn_lstm"]

    # Train model
    print("\n[5/5] Training model...")

    if is_deep_learning:
        # Deep learning models need full features (not split)
        features = extractor.extract(df)
        if args.use_lookahead:
            labels = labeler.label_with_lookahead(df)
        else:
            labels = labeler.label(df)

        # Create deep learning model
        deep_model_params = {
            "seq_length": args.seq_length,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "n_classes": args.n_classes,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "random_state": args.random_state,
        }

        if args.model == "lstm":
            model = LSTMClassifier(**deep_model_params)
        elif args.model == "transformer":
            model = TransformerClassifier(**deep_model_params)
        elif args.model == "cnn_lstm":
            model = CNNLSTMClassifier(**deep_model_params)

        print(f"Model: {args.model.upper()}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Hidden size: {args.hidden_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max epochs: {args.epochs}")

        model.fit(features, labels, eval_split=args.test_size, verbose=True)

        # Evaluate on test set (deep learning handles split internally)
        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)

        # Get predictions on last portion of data
        n_test = int(len(features) * args.test_size)
        test_features = features.iloc[-n_test:]
        test_labels = labels.iloc[-n_test:]

        y_pred = model.predict(test_features)
        # Align predictions with labels (accounting for seq_length)
        valid_test_labels = test_labels.iloc[args.seq_length - 1:]

        if len(y_pred) > 0 and len(valid_test_labels) > 0:
            test_acc = (y_pred == valid_test_labels).mean()
            print(f"Test accuracy: {test_acc:.4f}")

            from sklearn.metrics import classification_report
            print("\nClassification Report (Test Set):")
            print(classification_report(valid_test_labels, y_pred))

    else:
        # Traditional ML models
        if args.model == "ensemble":
            model = EnsembleClassifier(
                model_types=args.ensemble_models,
                voting="soft",
                random_state=args.random_state,
            )
        else:
            model = RegimeClassifier(
                model_type=args.model,
                random_state=args.random_state,
            )

        model.fit(X_train, y_train, verbose=True)

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)

        y_pred = model.predict(X_test)
        test_acc = (y_pred == y_test).mean()
        print(f"Test accuracy: {test_acc:.4f}")

        # Show per-class performance
        from sklearn.metrics import classification_report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred))

        # Cross-validation (if single model)
        if args.model != "ensemble" and args.cv_folds > 1:
            print("\nCross-Validation:")
            # Combine train and test for CV
            X_all = X_train._append(X_test)
            y_all = y_train._append(y_test)
            cv_results = model.cross_validate(X_all, y_all, cv=args.cv_folds)
            print(f"CV Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")

        # Feature importance
        if args.model != "ensemble":
            print("\nTop 10 Feature Importance:")
            importance = model.get_feature_importance()
            for i, (feat, imp) in enumerate(importance.head(10).items()):
                print(f"  {i+1}. {feat}: {imp:.4f}")

    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use .pt extension for deep learning models
    if is_deep_learning:
        model_filename = f"regime_classifier_{args.model}_{args.version}.pt"
    else:
        model_filename = f"regime_classifier_{args.version}.pkl"
    model_path = output_dir / model_filename

    model.save(model_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"\nTo upload to GCS, run:")
    print(f"  python upload.py --model {model_path}")


if __name__ == "__main__":
    main()
