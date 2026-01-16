#!/usr/bin/env python
"""Train supervised + deep learning ensemble (without unsupervised models).

This script:
1. Trains supervised models (RandomForest, XGBoost) via walk-forward validation
2. Loads pre-trained deep learning models (LSTM, Transformer, CNN-LSTM)
3. Creates an ensemble combining supervised and deep learning models

Usage:
    python train_supervised_dl_ensemble.py
    python train_supervised_dl_ensemble.py --n-classes 2
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import joblib

warnings.filterwarnings("ignore")

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import RegimeClassifier
from src.models.deep import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier
from src.validation.walk_forward import evaluate_supervised_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised + DL ensemble")
    parser.add_argument("--data", type=str, default="data/BTC.parquet", help="Path to OHLCV data")
    parser.add_argument("--n-classes", type=int, default=2, help="Number of regime classes")
    parser.add_argument("--train-size", type=int, default=500, help="Training window size")
    parser.add_argument("--test-size", type=int, default=60, help="Test window size")
    parser.add_argument("--step-size", type=int, default=60, help="Step size for validation")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    return parser.parse_args()


def train_supervised_models(features, labels, args):
    """Train and evaluate supervised models using walk-forward validation."""
    print("\n" + "=" * 70)
    print("SUPERVISED MODELS (Walk-Forward Validation)")
    print("=" * 70)

    results = {}
    model_types = ["random_forest", "xgboost"]

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


def load_deep_learning_models(model_dir: Path, n_classes: int):
    """Load pre-trained deep learning models."""
    print("\n" + "=" * 70)
    print("LOADING DEEP LEARNING MODELS")
    print("=" * 70)

    dl_models = {}

    # Try to load v2 models first (trained with 2 classes), then v1
    model_configs = [
        ("lstm", LSTMClassifier, f"regime_classifier_lstm_v2.pt"),
        ("transformer", TransformerClassifier, f"regime_classifier_transformer_v2.pt"),
        ("cnn_lstm", CNNLSTMClassifier, f"regime_classifier_cnn_lstm_v2.pt"),
    ]

    for name, model_class, filename in model_configs:
        model_path = model_dir / filename
        if model_path.exists():
            try:
                model = model_class.load(model_path)
                dl_models[name] = model
                print(f"  Loaded {name}: {model_path}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
        else:
            # Try v1
            v1_path = model_dir / filename.replace("_v2", "_v1")
            if v1_path.exists():
                try:
                    model = model_class.load(v1_path)
                    dl_models[name] = model
                    print(f"  Loaded {name}: {v1_path}")
                except Exception as e:
                    print(f"  Failed to load {name}: {e}")
            else:
                print(f"  Not found: {name}")

    return dl_models


class SupervisedDLEnsemble:
    """Ensemble combining supervised and deep learning models."""

    def __init__(
        self,
        n_classes: int = 2,
        supervised_models: list = None,
        dl_models: list = None,
        supervised_weight: float = 0.5,
        random_state: int = 42,
    ):
        self.n_classes = n_classes
        self.supervised_model_names = supervised_models or ["random_forest", "xgboost"]
        self.dl_model_names = dl_models or ["lstm", "transformer", "cnn_lstm"]
        self.supervised_weight = supervised_weight
        self.random_state = random_state

        self.supervised_models = {}
        self.dl_models = {}
        self.model_weights = {}
        self.is_fitted = False
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dl_models: dict = None,
        eval_split: float = 0.2,
        verbose: bool = True,
    ):
        """Train supervised models and set up DL models."""
        self.feature_names = list(X.columns)

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING SUPERVISED MODELS")
            print("=" * 60)

        # Train supervised models
        for i, model_name in enumerate(self.supervised_model_names):
            if verbose:
                print(f"\n[{i+1}/{len(self.supervised_model_names)}] Training {model_name}...")

            model = RegimeClassifier(
                model_type=model_name,
                scale_features=True,
                random_state=self.random_state,
            )
            model.fit(X, y, eval_split=eval_split, verbose=verbose)
            self.supervised_models[model_name] = model

        # Set DL models
        if dl_models:
            self.dl_models = dl_models
            if verbose:
                print(f"\n[DL Models] Loaded: {list(dl_models.keys())}")

        # Calculate weights
        self._calculate_weights()

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("ENSEMBLE TRAINING COMPLETE")
            print("=" * 60)
            print(f"Supervised models: {list(self.supervised_models.keys())}")
            print(f"Deep Learning models: {list(self.dl_models.keys())}")
            print(f"Model weights: {self.model_weights}")

        return self

    def _calculate_weights(self):
        """Calculate model weights based on configuration."""
        n_supervised = len(self.supervised_models)
        n_dl = len(self.dl_models)

        if n_supervised + n_dl == 0:
            return

        # Supervised weight per model
        sup_weight_each = self.supervised_weight / max(n_supervised, 1)
        # DL weight per model
        dl_weight_each = (1 - self.supervised_weight) / max(n_dl, 1)

        for name in self.supervised_models:
            self.model_weights[f"supervised_{name}"] = sup_weight_each

        for name in self.dl_models:
            self.model_weights[f"dl_{name}"] = dl_weight_each

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using weighted soft voting."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        proba_df = self.predict_proba(X)
        predictions = proba_df.idxmax(axis=1)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities using weighted average."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        all_probs = []
        weights = []

        # Supervised models
        for name, model in self.supervised_models.items():
            try:
                proba = model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index)
                all_probs.append(proba)
                weights.append(self.model_weights[f"supervised_{name}"])
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")

        # DL models
        for name, model in self.dl_models.items():
            try:
                proba = model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index)
                all_probs.append(proba)
                weights.append(self.model_weights[f"dl_{name}"])
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")

        if not all_probs:
            raise RuntimeError("No models produced predictions")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average of probabilities
        # Ensure all DataFrames have the same columns
        all_columns = set()
        for p in all_probs:
            all_columns.update(p.columns)
        all_columns = sorted(all_columns)

        aligned_probs = []
        for p in all_probs:
            aligned = p.reindex(columns=all_columns, fill_value=0.0)
            aligned_probs.append(aligned)

        weighted_proba = sum(w * p.values for w, p in zip(weights, aligned_probs))
        result = pd.DataFrame(weighted_proba, index=X.index, columns=all_columns)

        return result

    def save(self, path):
        """Save ensemble to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_classes": self.n_classes,
            "supervised_model_names": self.supervised_model_names,
            "dl_model_names": self.dl_model_names,
            "supervised_weight": self.supervised_weight,
            "random_state": self.random_state,
            "supervised_models": self.supervised_models,
            "model_weights": self.model_weights,
            "feature_names": self.feature_names,
        }

        joblib.dump(data, path)
        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path, dl_models: dict = None):
        """Load ensemble from file."""
        data = joblib.load(path)

        instance = cls(
            n_classes=data["n_classes"],
            supervised_models=data["supervised_model_names"],
            dl_models=data["dl_model_names"],
            supervised_weight=data["supervised_weight"],
            random_state=data["random_state"],
        )

        instance.supervised_models = data["supervised_models"]
        instance.model_weights = data["model_weights"]
        instance.feature_names = data["feature_names"]
        instance.dl_models = dl_models or {}
        instance.is_fitted = True

        return instance


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

    # Generate labels
    print("\n" + "=" * 70)
    print("GENERATING LABELS")
    print("=" * 70)
    labeler = RegimeLabeler(n_classes=args.n_classes)
    labels = labeler.label(df)
    print(f"Label distribution:")
    print(labels.value_counts())

    # Train supervised models with walk-forward validation
    sup_results = train_supervised_models(features, labels, args)

    # Load pre-trained deep learning models
    dl_models = load_deep_learning_models(Path(args.output_dir), args.n_classes)

    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    rows = []
    for name, result in sup_results.items():
        if result["n_folds"] > 0:
            rows.append({
                "Model": name.upper(),
                "Type": "Supervised",
                "Folds": result["n_folds"],
                "Accuracy": f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}",
                "F1": f"{result['mean_f1']:.4f} ± {result['std_f1']:.4f}",
            })

    for name in dl_models.keys():
        rows.append({
            "Model": name.upper(),
            "Type": "Deep Learning",
            "Folds": "pre-trained",
            "Accuracy": "N/A",
            "F1": "N/A",
        })

    if rows:
        comparison_df = pd.DataFrame(rows)
        print(comparison_df.to_string(index=False))

    # Train ensemble
    print("\n" + "=" * 70)
    print("TRAINING SUPERVISED + DL ENSEMBLE")
    print("=" * 70)

    ensemble = SupervisedDLEnsemble(
        n_classes=args.n_classes,
        supervised_models=["random_forest", "xgboost"],
        dl_models=list(dl_models.keys()),
        supervised_weight=0.5,  # 50% supervised, 50% DL
        random_state=42,
    )

    ensemble.fit(features, labels, dl_models=dl_models, eval_split=0.2, verbose=True)

    # Test prediction
    print("\n" + "=" * 70)
    print("TESTING ENSEMBLE PREDICTION")
    print("=" * 70)

    test_features = features.tail(10)
    predictions = ensemble.predict(test_features)
    proba = ensemble.predict_proba(test_features)

    print(f"Test predictions: {predictions.values}")
    print(f"\nProbabilities:")
    print(proba)

    # Save ensemble
    output_path = Path(args.output_dir) / "supervised_dl_ensemble.joblib"
    ensemble.save(output_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Ensemble saved to: {output_path}")


if __name__ == "__main__":
    main()
