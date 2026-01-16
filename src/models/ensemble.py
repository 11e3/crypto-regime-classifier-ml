"""Ensemble methods for regime classification."""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.classifier import RegimeClassifier


class EnsembleClassifier:
    """Ensemble classifier combining multiple models.

    Supports:
    - Voting (hard/soft)
    - Stacking (optional)

    Usage:
        ensemble = EnsembleClassifier(
            model_types=["random_forest", "gradient_boosting", "logistic"],
            voting="soft"
        )
        ensemble.fit(features, labels)
        predictions = ensemble.predict(new_features)
    """

    def __init__(
        self,
        model_types: list[str] = None,
        voting: str = "soft",
        weights: list[float] = None,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        """Initialize the ensemble classifier.

        Args:
            model_types: List of model types to include in ensemble
            voting: Voting strategy ('hard' or 'soft')
            weights: Optional weights for each model
            scale_features: Whether to scale features
            random_state: Random seed
        """
        self.model_types = model_types or ["random_forest", "gradient_boosting"]
        self.voting = voting
        self.weights = weights
        self.scale_features = scale_features
        self.random_state = random_state

        if weights and len(weights) != len(self.model_types):
            raise ValueError("Weights must match number of models")

        self.models: list[RegimeClassifier] = []
        self.feature_names: Optional[list[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_split: float = 0.2,
        verbose: bool = True,
    ) -> "EnsembleClassifier":
        """Train all models in the ensemble.

        Args:
            X: Feature DataFrame
            y: Labels Series
            eval_split: Fraction of data for evaluation
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        self.classes_ = np.unique(y)

        # Train each model
        self.models = []
        for i, model_type in enumerate(self.model_types):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training model {i+1}/{len(self.model_types)}: {model_type}")
                print("=" * 50)

            model = RegimeClassifier(
                model_type=model_type,
                scale_features=self.scale_features,
                random_state=self.random_state + i,
            )
            model.fit(X, y, eval_split=eval_split, verbose=verbose)
            self.models.append(model)

        self.is_fitted = True

        if verbose:
            print(f"\n{'='*50}")
            print("Ensemble training complete!")
            print(f"Models: {self.model_types}")
            print(f"Voting: {self.voting}")
            print("=" * 50)

        return self

    def predict(self, X: pd.DataFrame) -> Union[str, pd.Series]:
        """Predict regime using ensemble voting.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted regime(s)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        if self.voting == "hard":
            return self._hard_vote(X)
        else:
            return self._soft_vote(X)

    def _hard_vote(self, X: pd.DataFrame) -> Union[str, pd.Series]:
        """Hard voting: majority wins."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            if isinstance(pred, str):
                pred = pd.Series([pred], index=X.index)
            predictions.append(pred)

        # Stack predictions
        pred_df = pd.DataFrame(predictions).T

        # Weighted voting
        if self.weights:
            # For each row, compute weighted mode
            def weighted_mode(row):
                counts = {}
                for i, val in enumerate(row):
                    counts[val] = counts.get(val, 0) + self.weights[i]
                return max(counts, key=counts.get)

            result = pred_df.apply(weighted_mode, axis=1)
        else:
            # Simple majority
            result = pred_df.mode(axis=1)[0]

        if len(result) == 1:
            return result.iloc[0]
        return result

    def _soft_vote(self, X: pd.DataFrame) -> Union[str, pd.Series]:
        """Soft voting: average probabilities."""
        proba_list = []

        for model in self.models:
            proba = model.predict_proba(X)
            if isinstance(proba, dict):
                # Single prediction - convert to DataFrame
                proba = pd.DataFrame([proba], index=X.index)
            proba_list.append(proba)

        # Average probabilities (with optional weights)
        if self.weights:
            weighted_proba = sum(
                p * w for p, w in zip(proba_list, self.weights)
            ) / sum(self.weights)
        else:
            weighted_proba = sum(proba_list) / len(proba_list)

        # Get class with highest probability
        result = weighted_proba.idxmax(axis=1)

        if len(result) == 1:
            return result.iloc[0]
        return result

    def predict_proba(self, X: pd.DataFrame) -> dict:
        """Predict probability for each regime.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary or DataFrame with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        proba_list = []
        for model in self.models:
            proba = model.predict_proba(X)
            if isinstance(proba, dict):
                proba = pd.DataFrame([proba], index=X.index)
            proba_list.append(proba)

        # Average probabilities
        if self.weights:
            avg_proba = sum(
                p * w for p, w in zip(proba_list, self.weights)
            ) / sum(self.weights)
        else:
            avg_proba = sum(proba_list) / len(proba_list)

        if len(avg_proba) == 1:
            return avg_proba.iloc[0].to_dict()
        return avg_proba

    def get_feature_names(self) -> list[str]:
        """Return required feature names."""
        if self.feature_names is None:
            raise RuntimeError("Ensemble must be fitted to get feature names")
        return self.feature_names

    def get_model_agreements(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get predictions from each model for comparison.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with predictions from each model
        """
        predictions = {}
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            if isinstance(pred, str):
                pred = pd.Series([pred], index=X.index)
            predictions[f"{model.model_type}_{i}"] = pred

        return pd.DataFrame(predictions)

    def save(self, path: Union[str, Path]):
        """Save ensemble to file."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ensemble_data = {
            "models": self.models,
            "model_types": self.model_types,
            "voting": self.voting,
            "weights": self.weights,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "scale_features": self.scale_features,
            "random_state": self.random_state,
        }

        joblib.dump(ensemble_data, path)
        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnsembleClassifier":
        """Load ensemble from file."""
        data = joblib.load(path)

        instance = cls(
            model_types=data["model_types"],
            voting=data["voting"],
            weights=data["weights"],
            scale_features=data["scale_features"],
            random_state=data["random_state"],
        )

        instance.models = data["models"]
        instance.feature_names = data["feature_names"]
        instance.classes_ = data["classes_"]
        instance.is_fitted = True

        return instance
