"""Base class for deep learning regime classifiers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .dataset import RegimeDataset


class DeepRegimeClassifier(ABC):
    """Abstract base class for deep learning regime classifiers.

    Provides a consistent interface matching the sklearn-based RegimeClassifier:
    - fit(X, y): Train the model
    - predict(X): Get predictions
    - predict_proba(X): Get probability distributions
    - save(path): Save model to file
    - load(path): Load model from file

    Subclasses must implement:
    - _build_model(): Create the PyTorch model
    """

    def __init__(
        self,
        seq_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """Initialize the deep regime classifier.

        Args:
            seq_length: Number of time steps in input sequences
            hidden_size: Hidden dimension size
            num_layers: Number of layers in the model
            dropout: Dropout rate
            n_classes: Number of output classes (3 or 4)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device to use ('cuda', 'cpu', or None for auto)
            random_state: Random seed for reproducibility
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Will be set during training
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list[str]] = None
        self.n_features: Optional[int] = None
        self.is_fitted: bool = False
        self.training_history: list[dict] = []

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build and return the PyTorch model.

        Returns:
            PyTorch model instance
        """
        pass

    @property
    def model_type(self) -> str:
        """Return the model type name."""
        return self.__class__.__name__.replace("Classifier", "").lower()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_split: float = 0.15,
        verbose: bool = True,
    ) -> "DeepRegimeClassifier":
        """Train the classifier.

        Args:
            X: Feature DataFrame
            y: Labels Series
            eval_split: Fraction of data to use for validation
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        from .trainer import Trainer

        # Store feature info
        self.feature_names = list(X.columns)
        self.n_features = len(self.feature_names)

        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)

        # Prepare data
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        n_samples = len(X)
        train_end = int(n_samples * (1 - eval_split))

        train_X = X.iloc[:train_end]
        train_y = y.iloc[:train_end]
        val_X = X.iloc[train_end:]
        val_y = y.iloc[train_end:]

        # Create datasets
        train_dataset = RegimeDataset(
            train_X, train_y, seq_length=self.seq_length, fit_scaler=True, n_classes=self.n_classes
        )
        self.scaler = train_dataset.scaler

        val_dataset = RegimeDataset(
            val_X, val_y, seq_length=self.seq_length, scaler=self.scaler, fit_scaler=False, n_classes=self.n_classes
        )

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Get class weights
        class_weights = train_dataset.get_class_weights().to(self.device)

        # Create trainer and train
        trainer = Trainer(
            model=self.model,
            device=self.device,
            learning_rate=self.learning_rate,
            class_weights=class_weights,
        )

        self.training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            patience=self.patience,
            verbose=verbose,
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> Union[str, pd.Series]:
        """Predict regime for given features.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted regime(s) - string if single row, Series if multiple
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_features(X)
        proba = self._predict_proba_internal(X)

        # Get class with highest probability
        predictions_idx = proba.argmax(axis=1)
        predictions = [RegimeDataset.get_label_name(idx, self.n_classes) for idx in predictions_idx]

        if len(predictions) == 1:
            return predictions[0]

        # Adjust index to account for sequence length
        valid_idx = X.index[self.seq_length - 1 :]
        return pd.Series(predictions, index=valid_idx)

    def predict_proba(self, X: pd.DataFrame) -> dict:
        """Predict probability for each regime.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary with probability for each regime (single row)
            or DataFrame with probabilities (multiple rows)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_features(X)
        proba = self._predict_proba_internal(X)

        # Get class names (must be sorted by class index to match probability order)
        if self.n_classes == 2:
            label_map = RegimeDataset.LABEL_MAP_2CLASS
        elif self.n_classes == 3:
            label_map = RegimeDataset.LABEL_MAP_3CLASS
        else:
            label_map = RegimeDataset.LABEL_MAP
        # Sort by class index so column order matches probability array order
        class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

        if len(proba) == 1:
            return dict(zip(class_names, proba[0]))

        # Adjust index to account for sequence length
        valid_idx = X.index[self.seq_length - 1 :]
        return pd.DataFrame(proba, index=valid_idx, columns=class_names)

    def _predict_proba_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Internal method to get prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            NumPy array of shape (n_samples, n_classes)
        """
        self.model.eval()

        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_names].values)

        # Create sequences
        sequences = []
        for i in range(self.seq_length - 1, len(X_scaled)):
            seq = X_scaled[i - self.seq_length + 1 : i + 1]
            sequences.append(seq)

        if not sequences:
            raise ValueError(f"Input must have at least {self.seq_length} samples")

        sequences = np.array(sequences)
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(self.device)

        # Predict in batches
        all_proba = []
        with torch.no_grad():
            for i in range(0, len(sequences_tensor), self.batch_size):
                batch = sequences_tensor[i : i + self.batch_size]
                outputs = self.model(batch)
                proba = torch.softmax(outputs, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.concatenate(all_proba, axis=0)

    def _validate_features(self, X: pd.DataFrame):
        """Validate that input has required features."""
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def get_feature_names(self) -> list[str]:
        """Return required feature names for prediction."""
        if self.feature_names is None:
            raise RuntimeError("Model must be fitted to get feature names")
        return self.feature_names

    def save(self, path: Union[str, Path]):
        """Save model to file.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "config": {
                "seq_length": self.seq_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "n_classes": self.n_classes,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "patience": self.patience,
                "random_state": self.random_state,
            },
            "model_type": self.model_type,
            "training_history": self.training_history,
        }

        torch.save(model_data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeepRegimeClassifier":
        """Load model from file.

        Args:
            path: Path to the saved model

        Returns:
            Loaded DeepRegimeClassifier instance
        """
        model_data = torch.load(path, map_location="cpu", weights_only=False)

        config = model_data["config"]
        instance = cls(**config)

        instance.feature_names = model_data["feature_names"]
        instance.n_features = model_data["n_features"]
        instance.scaler = model_data["scaler"]
        instance.training_history = model_data.get("training_history", [])

        # Build and load model
        instance.model = instance._build_model()
        instance.model.load_state_dict(model_data["model_state_dict"])
        instance.model = instance.model.to(instance.device)
        instance.model.eval()

        instance.is_fitted = True

        return instance
