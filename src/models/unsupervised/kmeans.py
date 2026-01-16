"""K-Means clustering for regime classification."""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansClassifier:
    """K-Means clustering classifier for market regime detection.

    K-Means groups similar market conditions based on feature distributions:
    - Simple and fast
    - Works well when regimes have distinct characteristics
    - No temporal modeling (treats each point independently)

    Usage:
        model = KMeansClassifier(n_clusters=3)
        model.fit(features)
        predictions = model.predict(new_features)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        scale_features: bool = True,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        """Initialize K-Means classifier.

        Args:
            n_clusters: Number of clusters (regimes)
            scale_features: Whether to scale features
            random_state: Random seed
            n_init: Number of initializations
            max_iter: Maximum iterations
        """
        self.n_clusters = n_clusters
        self.scale_features = scale_features
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names: Optional[list[str]] = None
        self.cluster_labels: Optional[dict[int, str]] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> "KMeansClassifier":
        """Train K-Means clustering.

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
            print(f"Training K-Means with {self.n_clusters} clusters...")
            print(f"Training samples: {len(X_scaled)}")

        # Fit K-Means
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Map clusters to regime labels
        self._map_clusters_to_regimes(X_scaled, verbose=verbose)

        if verbose:
            print(f"\nK-Means training complete!")
            print(f"Inertia (within-cluster sum of squares): {self.model.inertia_:.2f}")

        return self

    def _map_clusters_to_regimes(self, X_scaled: np.ndarray, verbose: bool = True):
        """Map clusters to interpretable regime labels.

        Uses cluster center characteristics to determine regime type.
        """
        # Get cluster centers (first feature assumed to be return-related)
        centers = self.model.cluster_centers_
        center_returns = centers[:, 0]

        # Sort clusters by center return value
        sorted_clusters = np.argsort(center_returns)

        # Map to regime labels
        if self.n_clusters == 2:
            self.cluster_labels = {
                sorted_clusters[0]: "BEAR_TREND",
                sorted_clusters[1]: "BULL_TREND",
            }
        elif self.n_clusters == 3:
            self.cluster_labels = {
                sorted_clusters[0]: "BEAR_TREND",
                sorted_clusters[1]: "RANGING",
                sorted_clusters[2]: "BULL_TREND",
            }
        else:
            self.cluster_labels = {
                sorted_clusters[i]: f"REGIME_{i}" for i in range(self.n_clusters)
            }

        if verbose:
            predictions = self.model.predict(X_scaled)
            print(f"\nCluster mapping:")
            for cluster, label in self.cluster_labels.items():
                count = (predictions == cluster).sum()
                pct = count / len(predictions) * 100
                print(f"  Cluster {cluster} -> {label}: {count} samples ({pct:.1f}%)")

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

        # Predict clusters
        clusters = self.model.predict(X_scaled)

        # Map to regime labels
        predictions = [self.cluster_labels[c] for c in clusters]

        if len(predictions) == 1:
            return predictions[0]

        return pd.Series(predictions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> dict:
        """Predict probability for each regime (based on distance to centers).

        Uses inverse distance to cluster centers as pseudo-probability.

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

        # Calculate distances to each cluster center
        distances = self.model.transform(X_scaled)

        # Convert distances to probabilities (inverse distance, normalized)
        inv_distances = 1 / (distances + 1e-10)
        probs = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        # Map to regime labels
        regime_probs = {}
        for cluster, label in self.cluster_labels.items():
            regime_probs[label] = probs[:, cluster]

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

    def get_cluster_centers(self) -> pd.DataFrame:
        """Get cluster center coordinates."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")

        labels = [self.cluster_labels[i] for i in range(self.n_clusters)]
        return pd.DataFrame(
            self.model.cluster_centers_,
            index=labels,
            columns=self.feature_names,
        )

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
            "cluster_labels": self.cluster_labels,
            "n_clusters": self.n_clusters,
            "scale_features": self.scale_features,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
        }

        joblib.dump(model_data, path)
        print(f"K-Means model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KMeansClassifier":
        """Load model from file."""
        data = joblib.load(path)

        instance = cls(
            n_clusters=data["n_clusters"],
            scale_features=data["scale_features"],
            random_state=data["random_state"],
            n_init=data["n_init"],
            max_iter=data["max_iter"],
        )

        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.cluster_labels = data["cluster_labels"]
        instance.is_fitted = True

        return instance
