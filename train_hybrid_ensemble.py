#!/usr/bin/env python
"""Master training script for hybrid ensemble regime classifier.

This script trains and validates:
1. Supervised models (RandomForest, XGBoost, GradientBoosting) via walk-forward validation
2. Unsupervised models (HMM, K-Means, GMM) via expanding window validation
3. Deep learning models (LSTM, Transformer, CNN-LSTM) via walk-forward validation
4. Hybrid ensemble combining all models

Usage:
    python train_hybrid_ensemble.py
    python train_hybrid_ensemble.py --n-classes 3
    python train_hybrid_ensemble.py --skip-deep-learning
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import (
    RegimeClassifier,
    HybridEnsemble,
    HMMClassifier,
    KMeansClassifier,
    GMMClassifier,
)
from src.models.deep import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier
from src.validation import expanding_window_validation, walk_forward_validation
from src.validation.walk_forward import evaluate_supervised_model
from src.validation.expanding_window import evaluate_unsupervised_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train hybrid ensemble regime classifier")
    parser.add_argument("--data", type=str, default="data/BTC.parquet", help="Path to OHLCV data")
    parser.add_argument("--n-classes", type=int, default=2, help="Number of regime classes")
    parser.add_argument("--train-size", type=int, default=500, help="Training window size")
    parser.add_argument("--test-size", type=int, default=60, help="Test window size")
    parser.add_argument("--step-size", type=int, default=60, help="Step size for validation")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--skip-deep-learning", action="store_true", help="Skip deep learning models")
    parser.add_argument("--skip-unsupervised", action="store_true", help="Skip unsupervised models")
    parser.add_argument("--skip-supervised", action="store_true", help="Skip supervised models")
    return parser.parse_args()


def train_supervised_models(features, labels, args):
    """Train and evaluate supervised models using walk-forward validation."""
    print("\n" + "=" * 70)
    print("SUPERVISED MODELS (Walk-Forward Validation)")
    print("=" * 70)

    results = {}
    model_types = ["random_forest", "xgboost", "gradient_boosting", "logistic"]

    for model_type in model_types:
        result = evaluate_supervised_model(
            model_class=RegimeClassifier,
            model_params={"model_type": model_type, "scale_features": True, "random_state": 42},
            features=features,
            labels=labels,
            model_name=model_type.upper(),
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            is_deep_learning=False,
            verbose=True,
        )
        results[model_type] = result

    return results


def train_unsupervised_models(features, labels, args):
    """Train and evaluate unsupervised models using expanding window validation."""
    print("\n" + "=" * 70)
    print("UNSUPERVISED MODELS (Expanding Window Validation)")
    print("=" * 70)

    results = {}

    # HMM
    hmm_result = evaluate_unsupervised_model(
        model_class=HMMClassifier,
        model_params={"n_states": args.n_classes, "scale_features": True, "random_state": 42},
        features=features,
        labels=labels,
        model_name="HMM",
        initial_train_size=300,
        test_size=args.test_size,
        step_size=args.step_size,
        verbose=True,
    )
    results["hmm"] = hmm_result

    # K-Means
    kmeans_result = evaluate_unsupervised_model(
        model_class=KMeansClassifier,
        model_params={"n_clusters": args.n_classes, "scale_features": True, "random_state": 42},
        features=features,
        labels=labels,
        model_name="K-Means",
        initial_train_size=300,
        test_size=args.test_size,
        step_size=args.step_size,
        verbose=True,
    )
    results["kmeans"] = kmeans_result

    # GMM
    gmm_result = evaluate_unsupervised_model(
        model_class=GMMClassifier,
        model_params={"n_components": args.n_classes, "scale_features": True, "random_state": 42},
        features=features,
        labels=labels,
        model_name="GMM",
        initial_train_size=300,
        test_size=args.test_size,
        step_size=args.step_size,
        verbose=True,
    )
    results["gmm"] = gmm_result

    return results


def train_deep_learning_models(features, labels, args):
    """Train and evaluate deep learning models using walk-forward validation."""
    print("\n" + "=" * 70)
    print("DEEP LEARNING MODELS (Walk-Forward Validation)")
    print("=" * 70)

    results = {}

    dl_params = {
        "seq_length": 60,
        "hidden_size": 128,
        "num_layers": 2,
        "n_classes": args.n_classes,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "random_state": 42,
    }

    models = [
        ("LSTM", LSTMClassifier),
        ("Transformer", TransformerClassifier),
        ("CNN-LSTM", CNNLSTMClassifier),
    ]

    for name, model_class in models:
        result = evaluate_supervised_model(
            model_class=model_class,
            model_params=dl_params.copy(),
            features=features,
            labels=labels,
            model_name=name,
            train_size=args.train_size,
            test_size=120,  # Larger for deep learning due to seq_length
            step_size=90,
            is_deep_learning=True,
            verbose=True,
        )
        results[name.lower().replace("-", "_")] = result

    return results


def train_hybrid_ensemble(features, labels, args, all_results):
    """Train the hybrid ensemble using best performing models."""
    print("\n" + "=" * 70)
    print("TRAINING HYBRID ENSEMBLE")
    print("=" * 70)

    # Select best models based on validation performance
    supervised_models = ["random_forest", "xgboost"]
    unsupervised_models = ["hmm", "gmm"]

    # Calculate weights based on validation F1 scores
    sup_results = all_results.get("supervised", {})
    unsup_results = all_results.get("unsupervised", {})

    # Default weights if no results available
    supervised_weight = 0.6

    ensemble = HybridEnsemble(
        n_classes=args.n_classes,
        supervised_models=supervised_models,
        unsupervised_models=unsupervised_models,
        supervised_weight=supervised_weight,
        random_state=42,
    )

    # Train on all data
    ensemble.fit(features, labels, eval_split=0.2, verbose=True)

    # Save ensemble
    output_path = Path(args.output_dir) / "hybrid_ensemble.joblib"
    ensemble.save(output_path)

    return ensemble


def print_comparison_table(all_results):
    """Print comparison table of all models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    rows = []

    # Supervised models
    for name, result in all_results.get("supervised", {}).items():
        if result["n_folds"] > 0:
            rows.append({
                "Model": name.upper(),
                "Type": "Supervised",
                "Folds": result["n_folds"],
                "Accuracy": f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}",
                "F1": f"{result['mean_f1']:.4f} ± {result['std_f1']:.4f}",
            })

    # Unsupervised models
    for name, result in all_results.get("unsupervised", {}).items():
        if result["n_folds"] > 0:
            rows.append({
                "Model": name.upper(),
                "Type": "Unsupervised",
                "Folds": result["n_folds"],
                "Accuracy": f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}",
                "F1": f"{result['mean_f1']:.4f} ± {result['std_f1']:.4f}",
            })

    # Deep learning models
    for name, result in all_results.get("deep_learning", {}).items():
        if result["n_folds"] > 0:
            rows.append({
                "Model": name.upper(),
                "Type": "Deep Learning",
                "Folds": result["n_folds"],
                "Accuracy": f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}",
                "F1": f"{result['mean_f1']:.4f} ± {result['std_f1']:.4f}",
            })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    # Save results
    if rows:
        df.to_csv(Path("models") / "model_comparison.csv", index=False)
        print("\nResults saved to models/model_comparison.csv")


def main():
    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    df = load_ohlcv(args.data)
    print(f"Loaded {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")

    # Extract features
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)
    print(f"Features: {len(features.columns)} columns, {len(features)} rows")
    print(f"Feature columns: {list(features.columns[:10])}...")

    # Generate labels
    print("\n" + "=" * 70)
    print("GENERATING LABELS")
    print("=" * 70)
    labeler = RegimeLabeler(n_classes=args.n_classes)
    labels = labeler.label(df)
    print(f"Label distribution:")
    print(labels.value_counts())

    # Store all results
    all_results = {}

    # Train models
    if not args.skip_supervised:
        all_results["supervised"] = train_supervised_models(features, labels, args)

    if not args.skip_unsupervised:
        all_results["unsupervised"] = train_unsupervised_models(features, labels, args)

    if not args.skip_deep_learning:
        all_results["deep_learning"] = train_deep_learning_models(features, labels, args)

    # Print comparison
    print_comparison_table(all_results)

    # Train hybrid ensemble
    ensemble = train_hybrid_ensemble(features, labels, args, all_results)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Models saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
