"""Hidden Markov Model for regime classification."""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal


class HMMClassifier:
    """Hidden Markov Model classifier for market regime detection.

    HMM is the most classic approach for regime classification:
    - Models sequential dependencies between time steps
    - Probabilistic state transitions capture regime dynamics
    - Learns latent states from observed features

    Uses pomegranate library for HMM implementation.

    Usage:
        model = HMMClassifier(n_states=3)
        model.fit(features, labels=None)  # Unsupervised
        predictions = model.predict(new_features)
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        """Initialize HMM classifier.

        Args:
            n_states: Number of hidden states (regimes)
            covariance_type: Type of covariance (not used in pomegranate, kept for API compatibility)
            n_iter: Maximum iterations for EM algorithm
            scale_features: Whether to scale features
            random_state: Random seed
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.scale_features = scale_features
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names: Optional[list[str]] = None
        self.state_labels: Optional[dict[int, str]] = None
        self.is_fitted = False

        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> "HMMClassifier":
        """Train the HMM.

        Args:
            X: Feature DataFrame
            y: Optional labels for state mapping (not used in training)
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
            print(f"Training HMM with {self.n_states} states...")
            print(f"Training samples: {len(X_scaled)}")

        # Create distributions for each state with initial parameters
        n_features = X_scaled.shape[1]
        distributions = []

        # Initialize distributions with different means using k-means style initialization
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
        kmeans.fit(X_scaled)

        for i in range(self.n_states):
            # Use cluster centers as initial means
            means = torch.tensor(kmeans.cluster_centers_[i], dtype=torch.float32)
            # Use unit covariance as initial
            covs = torch.ones(n_features, dtype=torch.float32)
            dist = Normal(means=means, covs=covs, covariance_type="diag")
            distributions.append(dist)

        # Create HMM model
        self.model = DenseHMM(
            distributions=distributions,
            max_iter=self.n_iter,
            verbose=False,
        )

        # Fit HMM - pomegranate expects 3D input (batch, seq_len, features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        self.model.fit(X_tensor)
        self.is_fitted = True

        # Map states to regime labels based on feature characteristics
        self._map_states_to_regimes(X_scaled, verbose=verbose)

        if verbose:
            print(f"\nHMM training complete!")

        return self

    def _map_states_to_regimes(self, X_scaled: np.ndarray, verbose: bool = True):
        """Map HMM states to interpretable regime labels.

        Uses feature means to determine regime type:
        - High return features -> BULL_TREND
        - Low return features -> BEAR_TREND
        - Medium/volatile -> RANGING
        """
        # Get state predictions
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        state_predictions = self.model.predict(X_tensor).squeeze().numpy()

        # Calculate mean returns for each state (assuming first features relate to returns)
        state_means = {}
        for state in range(self.n_states):
            mask = state_predictions == state
            if mask.sum() > 0:
                state_means[state] = X_scaled[mask, 0].mean()  # Use first feature
            else:
                state_means[state] = 0.0

        # Sort states by mean return value
        sorted_states = sorted(state_means.items(), key=lambda x: x[1])

        # Map to regime labels
        if self.n_states == 2:
            self.state_labels = {
                sorted_states[0][0]: "BEAR_TREND",
                sorted_states[1][0]: "BULL_TREND",
            }
        elif self.n_states == 3:
            self.state_labels = {
                sorted_states[0][0]: "BEAR_TREND",
                sorted_states[1][0]: "RANGING",
                sorted_states[2][0]: "BULL_TREND",
            }
        else:
            self.state_labels = {
                state: f"REGIME_{i}" for i, (state, _) in enumerate(sorted_states)
            }

        if verbose:
            print(f"\nState mapping:")
            for state, label in self.state_labels.items():
                count = (state_predictions == state).sum()
                pct = count / len(state_predictions) * 100
                print(f"  State {state} -> {label}: {count} samples ({pct:.1f}%)")

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

        # Predict states
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        states = self.model.predict(X_tensor).squeeze().numpy()

        # Handle single prediction
        if states.ndim == 0:
            states = np.array([states.item()])

        # Map to regime labels
        predictions = [self.state_labels[int(s)] for s in states]

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

        # Get state probabilities
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        state_probs = self.model.predict_proba(X_tensor).squeeze().numpy()

        # Handle single prediction
        if state_probs.ndim == 1:
            state_probs = state_probs.reshape(1, -1)

        # Map to regime labels
        regime_probs = {}
        for state, label in self.state_labels.items():
            regime_probs[label] = state_probs[:, state]

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
            "state_labels": self.state_labels,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "scale_features": self.scale_features,
            "random_state": self.random_state,
        }

        joblib.dump(model_data, path)
        print(f"HMM model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HMMClassifier":
        """Load model from file."""
        data = joblib.load(path)

        instance = cls(
            n_states=data["n_states"],
            covariance_type=data["covariance_type"],
            n_iter=data["n_iter"],
            scale_features=data["scale_features"],
            random_state=data["random_state"],
        )

        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.state_labels = data["state_labels"]
        instance.is_fitted = True

        return instance
