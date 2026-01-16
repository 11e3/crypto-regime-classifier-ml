"""Hybrid ensemble combining supervised and unsupervised models."""

from pathlib import Path
from typing import Optional, Union, Any, List
import joblib
import numpy as np
import pandas as pd

from src.models.classifier import RegimeClassifier
from src.models.unsupervised import HMMClassifier, KMeansClassifier, GMMClassifier


class HybridEnsemble:
    """Hybrid ensemble combining supervised and unsupervised models.

    Architecture:
    - Supervised models: Trained via walk-forward validation
      - Random Forest, XGBoost, Gradient Boosting
    - Unsupervised models: Trained via expanding window
      - HMM, K-Means, GMM

    Ensemble Strategy:
    - Soft voting with learnable weights
    - Can weight by historical performance

    Usage:
        ensemble = HybridEnsemble(n_classes=2)
        ensemble.fit(features, labels)
        predictions = ensemble.predict(new_features)
    """

    def __init__(
        self,
        n_classes: int = 2,
        supervised_models: List[str] = None,
        unsupervised_models: List[str] = None,
        supervised_weight: float = 0.6,
        random_state: int = 42,
    ):
        """Initialize hybrid ensemble.

        Args:
            n_classes: Number of regime classes
            supervised_models: List of supervised model types
            unsupervised_models: List of unsupervised model types
            supervised_weight: Weight for supervised models (vs unsupervised)
            random_state: Random seed
        """
        self.n_classes = n_classes
        self.supervised_models = supervised_models or ["random_forest", "xgboost"]
        self.unsupervised_models = unsupervised_models or ["hmm", "gmm"]
        self.supervised_weight = supervised_weight
        self.random_state = random_state

        self._supervised: List[Any] = []
        self._unsupervised: List[Any] = []
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.model_weights: Optional[dict] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_split: float = 0.2,
        verbose: bool = True,
    ) -> "HybridEnsemble":
        """Train all models in the hybrid ensemble.

        Args:
            X: Feature DataFrame
            y: Labels Series
            eval_split: Fraction for evaluation (supervised only)
            verbose: Print progress

        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        self.classes_ = np.unique(y)

        # Initialize weights
        n_supervised = len(self.supervised_models)
        n_unsupervised = len(self.unsupervised_models)

        sup_weight_per_model = self.supervised_weight / n_supervised if n_supervised > 0 else 0
        unsup_weight_per_model = (1 - self.supervised_weight) / n_unsupervised if n_unsupervised > 0 else 0

        self.model_weights = {}

        # Train supervised models
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING SUPERVISED MODELS")
            print("=" * 60)

        self._supervised = []
        for i, model_type in enumerate(self.supervised_models):
            if verbose:
                print(f"\n[{i+1}/{n_supervised}] Training {model_type}...")

            model = RegimeClassifier(
                model_type=model_type,
                scale_features=True,
                random_state=self.random_state + i,
            )
            model.fit(X, y, eval_split=eval_split, verbose=verbose)
            self._supervised.append(model)
            self.model_weights[f"supervised_{model_type}"] = sup_weight_per_model

        # Train unsupervised models
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING UNSUPERVISED MODELS")
            print("=" * 60)

        self._unsupervised = []
        for i, model_type in enumerate(self.unsupervised_models):
            if verbose:
                print(f"\n[{i+1}/{n_unsupervised}] Training {model_type}...")

            if model_type == "hmm":
                model = HMMClassifier(
                    n_states=self.n_classes,
                    scale_features=True,
                    random_state=self.random_state + i + 100,
                )
            elif model_type == "kmeans":
                model = KMeansClassifier(
                    n_clusters=self.n_classes,
                    scale_features=True,
                    random_state=self.random_state + i + 100,
                )
            elif model_type == "gmm":
                model = GMMClassifier(
                    n_components=self.n_classes,
                    scale_features=True,
                    random_state=self.random_state + i + 100,
                )
            else:
                raise ValueError(f"Unknown unsupervised model: {model_type}")

            model.fit(X, verbose=verbose)
            self._unsupervised.append(model)
            self.model_weights[f"unsupervised_{model_type}"] = unsup_weight_per_model

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("HYBRID ENSEMBLE TRAINING COMPLETE")
            print("=" * 60)
            print(f"Supervised models: {self.supervised_models}")
            print(f"Unsupervised models: {self.unsupervised_models}")
            print(f"Model weights: {self.model_weights}")

        return self

    def predict(self, X: pd.DataFrame) -> Union[str, pd.Series]:
        """Predict regime using hybrid ensemble voting.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted regime(s)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Get probabilities from all models
        proba_list = []
        weight_list = []

        # Supervised model predictions
        for i, model in enumerate(self._supervised):
            try:
                proba = model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index)
                proba_list.append(proba)
                model_type = self.supervised_models[i]
                weight_list.append(self.model_weights[f"supervised_{model_type}"])
            except Exception as e:
                print(f"Warning: Supervised model {i} failed: {e}")

        # Unsupervised model predictions
        for i, model in enumerate(self._unsupervised):
            try:
                proba = model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index)

                # Ensure columns match classes
                for cls in self.classes_:
                    if cls not in proba.columns:
                        proba[cls] = 0.0

                proba = proba[sorted(proba.columns)]
                proba_list.append(proba)
                model_type = self.unsupervised_models[i]
                weight_list.append(self.model_weights[f"unsupervised_{model_type}"])
            except Exception as e:
                print(f"Warning: Unsupervised model {i} failed: {e}")

        if not proba_list:
            raise RuntimeError("All models failed to predict")

        # Weighted average of probabilities
        total_weight = sum(weight_list)
        weighted_proba = sum(
            p * w for p, w in zip(proba_list, weight_list)
        ) / total_weight

        # Get class with highest probability
        result = weighted_proba.idxmax(axis=1)

        if len(result) == 1:
            return result.iloc[0]
        return result

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probability for each regime.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        proba_list = []
        weight_list = []

        # Collect all probabilities
        for i, model in enumerate(self._supervised):
            try:
                proba = model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index)
                proba_list.append(proba)
                model_type = self.supervised_models[i]
                weight_list.append(self.model_weights[f"supervised_{model_type}"])
            except Exception:
                pass

        for i, model in enumerate(self._unsupervised):
            try:
                proba = model.predict_proba(X)
                if isinstance(proba, dict):
                    proba = pd.DataFrame([proba], index=X.index)

                for cls in self.classes_:
                    if cls not in proba.columns:
                        proba[cls] = 0.0

                proba = proba[sorted(proba.columns)]
                proba_list.append(proba)
                model_type = self.unsupervised_models[i]
                weight_list.append(self.model_weights[f"unsupervised_{model_type}"])
            except Exception:
                pass

        if not proba_list:
            raise RuntimeError("All models failed to predict")

        total_weight = sum(weight_list)
        avg_proba = sum(
            p * w for p, w in zip(proba_list, weight_list)
        ) / total_weight

        if len(avg_proba) == 1:
            return avg_proba.iloc[0].to_dict()
        return avg_proba

    def get_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get predictions from each individual model.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with predictions from each model
        """
        predictions = {}

        for i, model in enumerate(self._supervised):
            try:
                pred = model.predict(X)
                if isinstance(pred, str):
                    pred = pd.Series([pred], index=X.index)
                predictions[f"supervised_{self.supervised_models[i]}"] = pred
            except Exception:
                pass

        for i, model in enumerate(self._unsupervised):
            try:
                pred = model.predict(X)
                if isinstance(pred, str):
                    pred = pd.Series([pred], index=X.index)
                predictions[f"unsupervised_{self.unsupervised_models[i]}"] = pred
            except Exception:
                pass

        return pd.DataFrame(predictions)

    def get_feature_names(self) -> List[str]:
        """Return required feature names."""
        if self.feature_names is None:
            raise RuntimeError("Ensemble must be fitted to get feature names")
        return self.feature_names

    def update_weights(self, new_weights: dict):
        """Update model weights based on performance.

        Args:
            new_weights: Dictionary mapping model name to new weight
        """
        if self.model_weights is None:
            raise RuntimeError("Ensemble must be fitted first")

        for name, weight in new_weights.items():
            if name in self.model_weights:
                self.model_weights[name] = weight

        # Normalize weights
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in self.model_weights.items()}

    def save(self, path: Union[str, Path]):
        """Save ensemble to file."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "supervised": self._supervised,
            "unsupervised": self._unsupervised,
            "supervised_models": self.supervised_models,
            "unsupervised_models": self.unsupervised_models,
            "n_classes": self.n_classes,
            "supervised_weight": self.supervised_weight,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "model_weights": self.model_weights,
        }

        joblib.dump(data, path)
        print(f"Hybrid ensemble saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HybridEnsemble":
        """Load ensemble from file."""
        data = joblib.load(path)

        instance = cls(
            n_classes=data["n_classes"],
            supervised_models=data["supervised_models"],
            unsupervised_models=data["unsupervised_models"],
            supervised_weight=data["supervised_weight"],
            random_state=data["random_state"],
        )

        instance._supervised = data["supervised"]
        instance._unsupervised = data["unsupervised"]
        instance.feature_names = data["feature_names"]
        instance.classes_ = data["classes_"]
        instance.model_weights = data["model_weights"]
        instance.is_fitted = True

        return instance
