"""Ensemble of deep learning models for regime classification."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from .base import DeepRegimeClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier
from .cnn_lstm import CNNLSTMClassifier
from .dataset import RegimeDataset


class DeepEnsemble:
    """Ensemble of multiple deep learning models.

    Combines predictions from LSTM, Transformer, and CNN-LSTM models
    using averaging or voting strategies.

    Usage:
        # Load pre-trained models
        ensemble = DeepEnsemble.from_saved_models(
            lstm_path="models/lstm.pt",
            transformer_path="models/transformer.pt",
            cnn_lstm_path="models/cnn_lstm.pt",
        )

        # Predict
        predictions = ensemble.predict(features_df)
        probabilities = ensemble.predict_proba(features_df)
    """

    MODEL_CLASSES = {
        "lstm": LSTMClassifier,
        "transformer": TransformerClassifier,
        "cnn_lstm": CNNLSTMClassifier,
    }

    def __init__(
        self,
        models: list[DeepRegimeClassifier],
        weights: Optional[list[float]] = None,
        voting: str = "soft",
        threshold: Optional[float] = None,
    ):
        """Initialize the ensemble.

        Args:
            models: List of trained DeepRegimeClassifier instances
            weights: Optional weights for each model (must sum to 1)
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
            threshold: Optional prediction threshold for positive class (2-class only)
        """
        if not models:
            raise ValueError("At least one model required")

        self.models = models
        self.voting = voting
        self.threshold = threshold

        # Set weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # Get config from first model
        self.n_classes = models[0].n_classes
        self.seq_length = models[0].seq_length
        self.feature_names = models[0].feature_names

        # Validate all models have same config
        for model in models[1:]:
            if model.n_classes != self.n_classes:
                raise ValueError("All models must have same n_classes")
            if model.seq_length != self.seq_length:
                raise ValueError("All models must have same seq_length")

    @classmethod
    def from_saved_models(
        cls,
        lstm_path: Optional[Union[str, Path]] = None,
        transformer_path: Optional[Union[str, Path]] = None,
        cnn_lstm_path: Optional[Union[str, Path]] = None,
        weights: Optional[list[float]] = None,
        voting: str = "soft",
        threshold: Optional[float] = None,
    ) -> "DeepEnsemble":
        """Create ensemble from saved model files.

        Args:
            lstm_path: Path to saved LSTM model
            transformer_path: Path to saved Transformer model
            cnn_lstm_path: Path to saved CNN-LSTM model
            weights: Optional weights for each model
            voting: 'soft' or 'hard' voting
            threshold: Optional prediction threshold

        Returns:
            DeepEnsemble instance
        """
        models = []
        model_paths = [
            ("lstm", lstm_path),
            ("transformer", transformer_path),
            ("cnn_lstm", cnn_lstm_path),
        ]

        for model_type, path in model_paths:
            if path is not None:
                model_class = cls.MODEL_CLASSES[model_type]
                model = model_class.load(path)
                models.append(model)
                print(f"Loaded {model_type} model from {path}")

        if not models:
            raise ValueError("At least one model path must be provided")

        return cls(models, weights=weights, voting=voting, threshold=threshold)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get averaged probability predictions from all models.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with probabilities for each class
        """
        all_proba = []
        common_idx = None

        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            if isinstance(proba, dict):
                proba = pd.DataFrame([proba])
            all_proba.append((proba, weight))

            # Track common index
            if common_idx is None:
                common_idx = proba.index
            else:
                common_idx = common_idx.intersection(proba.index)

        # Align all probabilities to common index and average
        ensemble_proba = None
        for proba, weight in all_proba:
            aligned = proba.loc[common_idx] * weight
            if ensemble_proba is None:
                ensemble_proba = aligned
            else:
                ensemble_proba = ensemble_proba + aligned

        return ensemble_proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Get predictions from ensemble.

        Args:
            X: Feature DataFrame

        Returns:
            Series with predicted labels
        """
        if self.voting == "soft":
            proba = self.predict_proba(X)

            # For 2-class, optionally use custom threshold
            if self.n_classes == 2 and self.threshold is not None:
                # BULL_TREND is class 1 in 2-class
                bull_proba = proba["BULL_TREND"]
                predictions = ["BULL_TREND" if p >= self.threshold else "NOT_BULL" for p in bull_proba]
                return pd.Series(predictions, index=proba.index)

            # Otherwise use argmax
            predictions = proba.idxmax(axis=1)
            return predictions

        else:  # hard voting
            all_preds = []
            for model in self.models:
                preds = model.predict(X)
                if isinstance(preds, str):
                    preds = pd.Series([preds])
                all_preds.append(preds)

            # Align indices
            common_idx = all_preds[0].index
            for preds in all_preds[1:]:
                common_idx = common_idx.intersection(preds.index)

            # Majority vote
            votes = pd.DataFrame({f"model_{i}": preds.loc[common_idx] for i, preds in enumerate(all_preds)})
            predictions = votes.mode(axis=1)[0]
            return predictions

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get predictions with confidence scores and model agreement.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with columns: prediction, confidence, agreement, bull_proba
        """
        proba = self.predict_proba(X)

        # Get individual model predictions for agreement
        all_preds = []
        for model in self.models:
            preds = model.predict(X)
            if isinstance(preds, str):
                preds = pd.Series([preds], index=proba.index[:1])
            all_preds.append(preds)

        # Calculate agreement
        common_idx = proba.index
        for preds in all_preds:
            common_idx = common_idx.intersection(preds.index)

        votes = pd.DataFrame({f"model_{i}": preds.loc[common_idx] for i, preds in enumerate(all_preds)})

        # Agreement = fraction of models that agree with final prediction
        final_pred = proba.loc[common_idx].idxmax(axis=1)
        agreement = votes.apply(lambda row: (row == final_pred.loc[row.name]).sum() / len(self.models), axis=1)

        # Build result
        result = pd.DataFrame({
            "prediction": final_pred,
            "confidence": proba.loc[common_idx].max(axis=1),
            "agreement": agreement,
        }, index=common_idx)

        # Add BULL probability for 2-class
        if self.n_classes == 2:
            result["bull_proba"] = proba.loc[common_idx, "BULL_TREND"]

        return result

    def evaluate(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> dict:
        """Evaluate ensemble on test data.

        Args:
            X: Feature DataFrame
            y: True labels
            verbose: Whether to print results

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
            confusion_matrix,
        )

        predictions = self.predict(X)

        # Align with labels
        common_idx = predictions.index.intersection(y.index)
        y_true = y.loc[common_idx]
        y_pred = predictions.loc[common_idx]

        # Calculate metrics
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }

        # Per-class metrics for 2-class
        if self.n_classes == 2:
            results["precision_bull"] = precision_score(y_true, y_pred, pos_label="BULL_TREND")
            results["recall_bull"] = recall_score(y_true, y_pred, pos_label="BULL_TREND")
            results["f1_bull"] = f1_score(y_true, y_pred, pos_label="BULL_TREND")

        results["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        results["classification_report"] = classification_report(y_true, y_pred)

        if verbose:
            print(f"Ensemble Evaluation ({len(self.models)} models)")
            print("=" * 50)
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"F1 Score:  {results['f1']:.4f}")

            if self.n_classes == 2:
                print(f"\nBULL_TREND Metrics:")
                print(f"  Precision: {results['precision_bull']:.4f}")
                print(f"  Recall:    {results['recall_bull']:.4f}")
                print(f"  F1:        {results['f1_bull']:.4f}")

            print(f"\nClassification Report:\n{results['classification_report']}")

        return results

    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare individual model performance vs ensemble.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            DataFrame with metrics for each model and ensemble
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        results = []

        # Evaluate individual models
        for i, model in enumerate(self.models):
            preds = model.predict(X)
            if isinstance(preds, str):
                preds = pd.Series([preds])

            common_idx = preds.index.intersection(y.index)
            y_true = y.loc[common_idx]
            y_pred = preds.loc[common_idx]

            model_results = {
                "model": model.model_type,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1": f1_score(y_true, y_pred, average="weighted"),
            }

            if self.n_classes == 2:
                model_results["precision_bull"] = precision_score(y_true, y_pred, pos_label="BULL_TREND")
                model_results["recall_bull"] = recall_score(y_true, y_pred, pos_label="BULL_TREND")

            results.append(model_results)

        # Evaluate ensemble
        ensemble_preds = self.predict(X)
        common_idx = ensemble_preds.index.intersection(y.index)
        y_true = y.loc[common_idx]
        y_pred = ensemble_preds.loc[common_idx]

        ensemble_results = {
            "model": "ENSEMBLE",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }

        if self.n_classes == 2:
            ensemble_results["precision_bull"] = precision_score(y_true, y_pred, pos_label="BULL_TREND")
            ensemble_results["recall_bull"] = recall_score(y_true, y_pred, pos_label="BULL_TREND")

        results.append(ensemble_results)

        return pd.DataFrame(results)

    def save(self, path: Union[str, Path]):
        """Save ensemble configuration (model paths should be saved separately).

        Args:
            path: Path to save ensemble config
        """
        config = {
            "weights": self.weights,
            "voting": self.voting,
            "threshold": self.threshold,
            "n_classes": self.n_classes,
            "seq_length": self.seq_length,
            "n_models": len(self.models),
            "model_types": [m.model_type for m in self.models],
        }
        torch.save(config, path)
        print(f"Ensemble config saved to {path}")

    def __repr__(self) -> str:
        model_types = [m.model_type for m in self.models]
        return f"DeepEnsemble(models={model_types}, voting={self.voting})"
