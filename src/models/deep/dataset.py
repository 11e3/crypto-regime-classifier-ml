"""PyTorch Dataset for regime classification."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class RegimeDataset(Dataset):
    """PyTorch Dataset for sequence-based regime classification.

    Creates sequences of features for time-series modeling.
    Each sample is a sequence of `seq_length` time steps, with the label
    being the regime at the last time step.

    Usage:
        dataset = RegimeDataset(features_df, labels_series, seq_length=60)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    # Class label mapping
    LABEL_MAP = {
        "BULL_TREND": 0,
        "BEAR_TREND": 1,
        "SIDEWAYS": 2,
        "HIGH_VOL": 3,
    }

    LABEL_MAP_3CLASS = {
        "BULL_TREND": 0,
        "BEAR_TREND": 1,
        "SIDEWAYS": 2,
    }

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        seq_length: int = 60,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        n_classes: int = 3,
    ):
        """Initialize the dataset.

        Args:
            features: DataFrame with feature columns
            labels: Series with regime labels
            seq_length: Number of time steps in each sequence
            scaler: Optional pre-fitted scaler. If None, creates new one
            fit_scaler: Whether to fit the scaler on this data
            n_classes: Number of classes (3 or 4)
        """
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.label_map = self.LABEL_MAP_3CLASS if n_classes == 3 else self.LABEL_MAP

        # Align indices
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Store feature names
        self.feature_names = list(features.columns)
        self.n_features = len(self.feature_names)

        # Handle scaler
        if scaler is not None:
            self.scaler = scaler
            self.features_scaled = self.scaler.transform(features.values)
        elif fit_scaler:
            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(features.values)
        else:
            self.scaler = None
            self.features_scaled = features.values

        # Convert labels to numeric
        self.labels = np.array([self.label_map[l] for l in labels.values])

        # Calculate valid indices (need seq_length samples before each point)
        self.valid_indices = list(range(seq_length - 1, len(self.features_scaled)))

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (sequence, label) tensors
        """
        # Get the actual index in the data
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_length + 1

        # Extract sequence
        sequence = self.features_scaled[start_idx : end_idx + 1]

        # Get label (regime at the last time step)
        label = self.labels[end_idx]

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data.

        Returns:
            Tensor with weight for each class
        """
        # Count samples per class
        class_counts = np.bincount(self.labels, minlength=self.n_classes)

        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)

        # Inverse frequency weighting
        weights = len(self.labels) / (self.n_classes * class_counts)

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for weighted sampling.

        Returns:
            Tensor with weight for each sample
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = class_weights[self.labels[self.valid_indices]]
        return torch.tensor(sample_weights, dtype=torch.float32)

    @staticmethod
    def get_label_name(label_idx: int, n_classes: int = 3) -> str:
        """Convert numeric label back to string.

        Args:
            label_idx: Numeric label index
            n_classes: Number of classes

        Returns:
            String label name
        """
        label_map = RegimeDataset.LABEL_MAP_3CLASS if n_classes == 3 else RegimeDataset.LABEL_MAP
        reverse_map = {v: k for k, v in label_map.items()}
        return reverse_map.get(label_idx, "UNKNOWN")


def create_dataloaders(
    features: pd.DataFrame,
    labels: pd.Series,
    seq_length: int = 60,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    n_classes: int = 3,
    shuffle_train: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, StandardScaler]:
    """Create train, validation, and test dataloaders with time-based split.

    Args:
        features: DataFrame with feature columns
        labels: Series with regime labels
        seq_length: Number of time steps in each sequence
        batch_size: Batch size for dataloaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        n_classes: Number of classes (3 or 4)
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Align indices
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    n_samples = len(features)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Time-based split (no shuffle before split to prevent data leakage)
    train_features = features.iloc[:train_end]
    train_labels = labels.iloc[:train_end]

    val_features = features.iloc[train_end:val_end]
    val_labels = labels.iloc[train_end:val_end]

    test_features = features.iloc[val_end:]
    test_labels = labels.iloc[val_end:]

    # Create datasets (fit scaler only on training data)
    train_dataset = RegimeDataset(
        train_features, train_labels, seq_length=seq_length, fit_scaler=True, n_classes=n_classes
    )

    val_dataset = RegimeDataset(
        val_features, val_labels, seq_length=seq_length, scaler=train_dataset.scaler, fit_scaler=False, n_classes=n_classes
    )

    test_dataset = RegimeDataset(
        test_features, test_labels, seq_length=seq_length, scaler=train_dataset.scaler, fit_scaler=False, n_classes=n_classes
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.scaler
