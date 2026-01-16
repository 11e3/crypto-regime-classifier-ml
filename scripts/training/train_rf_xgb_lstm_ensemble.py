#!/usr/bin/env python
"""Train RF + XGBoost + LSTM ensemble.

This script:
1. Trains supervised models (RandomForest, XGBoost) via walk-forward validation
2. Loads pre-trained LSTM model
3. Creates an ensemble combining the three models

Usage:
    python train_rf_xgb_lstm_ensemble.py
    python train_rf_xgb_lstm_ensemble.py --n-classes 2
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
from src.models.deep import LSTMClassifier
from src.validation.walk_forward import evaluate_supervised_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train RF + XGBoost + LSTM ensemble")
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


def load_lstm_model(model_dir: Path):
    """Load pre-trained LSTM model."""
    print("\n" + "=" * 70)
    print("LOADING LSTM MODEL")
    print("=" * 70)

    # Try v2 first, then v1
    for version in ["v2", "v1"]:
        model_path = model_dir / f"regime_classifier_lstm_{version}.pt"
        if model_path.exists():
            try:
                model = LSTMClassifier.load(model_path)
                print(f"  Loaded LSTM: {model_path}")
                return model
            except Exception as e:
                print(f"  Failed to load LSTM {version}: {e}")

    print("  LSTM model not found!")
    return None


class RFXGBLSTMEnsemble:
    """Ensemble combining Random Forest, XGBoost, and LSTM."""

    def __init__(
        self,
        n_classes: int = 2,
        supervised_weight: float = 0.6,  # 60% supervised, 40% LSTM
        random_state: int = 42,
    ):
        self.n_classes = n_classes
        self.supervised_weight = supervised_weight
        self.random_state = random_state

        self.rf_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.model_weights = {}
        self.is_fitted = False
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        lstm_model=None,
        eval_split: float = 0.2,
        verbose: bool = True,
    ):
        """Train RF and XGBoost models, set LSTM model."""
        self.feature_names = list(X.columns)

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING RF + XGBOOST + LSTM ENSEMBLE")
            print("=" * 60)

        # Train Random Forest
        if verbose:
            print("\n[1/3] Training Random Forest...")
        self.rf_model = RegimeClassifier(
            model_type="random_forest",
            scale_features=True,
            random_state=self.random_state,
        )
        self.rf_model.fit(X, y, eval_split=eval_split, verbose=verbose)

        # Train XGBoost
        if verbose:
            print("\n[2/3] Training XGBoost...")
        self.xgb_model = RegimeClassifier(
            model_type="xgboost",
            scale_features=True,
            random_state=self.random_state,
        )
        self.xgb_model.fit(X, y, eval_split=eval_split, verbose=verbose)

        # Set LSTM model
        if lstm_model:
            self.lstm_model = lstm_model
            if verbose:
                print("\n[3/3] LSTM model loaded (pre-trained)")

        # Calculate weights
        self._calculate_weights()

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("ENSEMBLE TRAINING COMPLETE")
            print("=" * 60)
            print(f"Models: RF, XGBoost, LSTM")
            print(f"Weights: {self.model_weights}")

        return self

    def _calculate_weights(self):
        """Calculate model weights."""
        # Supervised weight split between RF and XGBoost
        sup_weight_each = self.supervised_weight / 2

        self.model_weights["rf"] = sup_weight_each
        self.model_weights["xgb"] = sup_weight_each

        if self.lstm_model:
            self.model_weights["lstm"] = 1 - self.supervised_weight
        else:
            # If no LSTM, redistribute to supervised
            self.model_weights["rf"] = 0.5
            self.model_weights["xgb"] = 0.5

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
        common_index = X.index

        # Random Forest
        try:
            proba = self.rf_model.predict_proba(X)
            if isinstance(proba, dict):
                proba = pd.DataFrame([proba], index=X.index)
            all_probs.append(("rf", proba))
            weights.append(self.model_weights["rf"])
        except Exception as e:
            print(f"Warning: RF prediction failed: {e}")

        # XGBoost
        try:
            proba = self.xgb_model.predict_proba(X)
            if isinstance(proba, dict):
                proba = pd.DataFrame([proba], index=X.index)
            all_probs.append(("xgb", proba))
            weights.append(self.model_weights["xgb"])
        except Exception as e:
            print(f"Warning: XGBoost prediction failed: {e}")

        # LSTM - has different output length due to seq_length
        if self.lstm_model and len(X) >= 60:
            try:
                proba = self.lstm_model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index[-1:])
                all_probs.append(("lstm", proba))
                weights.append(self.model_weights["lstm"])
                # Use LSTM's index as common (shorter)
                common_index = proba.index
            except Exception as e:
                print(f"Warning: LSTM prediction failed: {e}")

        if not all_probs:
            raise RuntimeError("No models produced predictions")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Get all columns
        all_columns = set()
        for name, p in all_probs:
            all_columns.update(p.columns)
        all_columns = sorted(all_columns)

        # Align all probabilities to common index
        aligned_probs = []
        for (name, p), w in zip(all_probs, weights):
            # Align to common index
            aligned = p.reindex(index=common_index, columns=all_columns, fill_value=0.0)
            # Forward fill for supervised models that have all predictions
            if name in ["rf", "xgb"] and len(p) > len(common_index):
                aligned = p.loc[common_index].reindex(columns=all_columns, fill_value=0.0)
            aligned_probs.append(aligned)

        weighted_proba = sum(w * p.values for w, p in zip(weights, aligned_probs))
        result = pd.DataFrame(weighted_proba, index=common_index, columns=all_columns)

        return result

    def save(self, path):
        """Save ensemble to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_classes": self.n_classes,
            "supervised_weight": self.supervised_weight,
            "random_state": self.random_state,
            "rf_model": self.rf_model,
            "xgb_model": self.xgb_model,
            "model_weights": self.model_weights,
            "feature_names": self.feature_names,
        }

        joblib.dump(data, path)
        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path, lstm_model=None):
        """Load ensemble from file."""
        data = joblib.load(path)

        instance = cls(
            n_classes=data["n_classes"],
            supervised_weight=data["supervised_weight"],
            random_state=data["random_state"],
        )

        instance.rf_model = data["rf_model"]
        instance.xgb_model = data["xgb_model"]
        instance.model_weights = data["model_weights"]
        instance.feature_names = data["feature_names"]
        instance.lstm_model = lstm_model
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

    # Load pre-trained LSTM model
    lstm_model = load_lstm_model(Path(args.output_dir))

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

    if lstm_model:
        rows.append({
            "Model": "LSTM",
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
    print("TRAINING RF + XGBOOST + LSTM ENSEMBLE")
    print("=" * 70)

    ensemble = RFXGBLSTMEnsemble(
        n_classes=args.n_classes,
        supervised_weight=0.6,  # 60% supervised (RF + XGB), 40% LSTM
        random_state=42,
    )

    ensemble.fit(features, labels, lstm_model=lstm_model, eval_split=0.2, verbose=True)

    # Test prediction
    print("\n" + "=" * 70)
    print("TESTING ENSEMBLE PREDICTION")
    print("=" * 70)

    test_features = features.tail(100)  # Use 100 samples for LSTM
    predictions = ensemble.predict(test_features)
    proba = ensemble.predict_proba(test_features)

    print(f"Test predictions (last 10): {predictions.tail(10).values}")
    print(f"\nProbabilities (last 5):")
    print(proba.tail(5))

    # Save ensemble
    output_path = Path(args.output_dir) / "rf_xgb_lstm_ensemble.joblib"
    ensemble.save(output_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Ensemble saved to: {output_path}")
    print(f"Model weights: RF={ensemble.model_weights['rf']:.2f}, XGB={ensemble.model_weights['xgb']:.2f}, LSTM={ensemble.model_weights.get('lstm', 0):.2f}")


if __name__ == "__main__":
    main()
