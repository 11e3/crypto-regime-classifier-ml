"""Gaussian Mixture Model for regime classification."""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMClassifier:
    """Gaussian Mixture Model classifier for market regime detection.

    GMM provides soft clustering with probabilistic assignments:
    - Assumes data comes from mixture of Gaussian distributions
    - Provides genuine probabilities (unlike K-Means)
    - Better handles overlapping regime boundaries

    Usage:
        model = GMMClassifier(n_components=3)
        model.fit(features)
        predictions = model.predict(new_features)
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        scale_features: bool = True,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 200,
    ):
        """Initialize GMM classifier.

        Args:
            n_components: Number of mixture components (regimes)
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            scale_features: Whether to scale features
            random_state: Random seed
            n_init: Number of initializations
            max_iter: Maximum EM iterations
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.scale_features = scale_features
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names: Optional[list[str]] = None
        self.component_labels: Optional[dict[int, str]] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> "GMMClassifier":
        """Train GMM.

        Args:
            X: Feature DataFrame
            y: Optional labels (not used in training)
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)

        # Handle missing values
        X_clean = X.dropna()

        # Scale features
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean.values

        if verbose:
            print(f"Training GMM with {self.n_components} components...")
            print(f"Training samples: {len(X_scaled)}")

        # Fit GMM
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Map components to regime labels
        self._map_components_to_regimes(X_scaled, verbose=verbose)

        if verbose:
            print(f"\nGMM training complete!")
            print(f"Converged: {self.model.converged_}")
            print(f"Lower bound (log-likelihood): {self.model.lower_bound_:.2f}")
            print(f"BIC: {self.model.bic(X_scaled):.2f}")
            print(f"AIC: {self.model.aic(X_scaled):.2f}")

        return self

    def _map_components_to_regimes(self, X_scaled: np.ndarray, verbose: bool = True):
        """Map GMM components to interpretable regime labels.

        Uses component means to determine regime type.
        """
        # Get component means (first feature assumed to be return-related)
        means = self.model.means_
        mean_returns = means[:, 0]

        # Sort components by mean return value
        sorted_components = np.argsort(mean_returns)

        # Map to regime labels
        if self.n_components == 2:
            self.component_labels = {
                sorted_components[0]: "BEAR_TREND",
                sorted_components[1]: "BULL_TREND",
            }
        elif self.n_components == 3:
            self.component_labels = {
                sorted_components[0]: "BEAR_TREND",
                sorted_components[1]: "RANGING",
                sorted_components[2]: "BULL_TREND",
            }
        else:
            self.component_labels = {
                sorted_components[i]: f"REGIME_{i}" for i in range(self.n_components)
            }

        if verbose:
            predictions = self.model.predict(X_scaled)
            print(f"\nComponent mapping:")
            for comp, label in self.component_labels.items():
                count = (predictions == comp).sum()
                pct = count / len(predictions) * 100
                weight = self.model.weights_[comp] * 100
                print(f"  Component {comp} -> {label}: {count} samples ({pct:.1f}%), weight: {weight:.1f}%")

    def predict(self, X: pd.DataFrame) -> Union[str, pd.Series]:
        """Predict regime for given features.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted regime(s)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_features(X)

        # Handle missing values
        X_clean = X.ffill().bfill()

        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X_clean[self.feature_names])
        else:
            X_scaled = X_clean[self.feature_names].values

        # Predict components
        components = self.model.predict(X_scaled)

        # Map to regime labels
        predictions = [self.component_labels[c] for c in components]

        if len(predictions) == 1:
            return predictions[0]

        return pd.Series(predictions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> dict:
        """Predict probability for each regime.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary or DataFrame with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_features(X)

        # Handle missing values
        X_clean = X.ffill().bfill()

        if self.scaler:
            X_scaled = self.scaler.transform(X_clean[self.feature_names])
        else:
            X_scaled = X_clean[self.feature_names].values

        # Get component probabilities
        probs = self.model.predict_proba(X_scaled)

        # Map to regime labels
        regime_probs = {}
        for comp, label in self.component_labels.items():
            regime_probs[label] = probs[:, comp]

        proba_df = pd.DataFrame(regime_probs, index=X.index)

        if len(proba_df) == 1:
            return proba_df.iloc[0].to_dict()

        return proba_df

    def _validate_features(self, X: pd.DataFrame):
        """Validate that input has required features."""
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def get_feature_names(self) -> list[str]:
        """Return required feature names."""
        if self.feature_names is None:
            raise RuntimeError("Model must be fitted to get feature names")
        return self.feature_names

    def get_component_params(self) -> dict:
        """Get GMM component parameters."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")

        labels = [self.component_labels[i] for i in range(self.n_components)]

        return {
            "means": pd.DataFrame(
                self.model.means_,
                index=labels,
                columns=self.feature_names,
            ),
            "weights": pd.Series(self.model.weights_, index=labels),
        }

    def save(self, path: Union[str, Path]):
        """Save model to file."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "component_labels": self.component_labels,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "scale_features": self.scale_features,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
        }

        joblib.dump(model_data, path)
        print(f"GMM model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GMMClassifier":
        """Load model from file."""
        data = joblib.load(path)

        instance = cls(
            n_components=data["n_components"],
            covariance_type=data["covariance_type"],
            scale_features=data["scale_features"],
            random_state=data["random_state"],
            n_init=data["n_init"],
            max_iter=data["max_iter"],
        )

        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.component_labels = data["component_labels"]
        instance.is_fitted = True

        return instance
