"""CNN-LSTM hybrid regime classifier."""

import torch
import torch.nn as nn

from .base import DeepRegimeClassifier


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for sequence classification.

    Architecture:
    - 1D CNN layers for local pattern extraction
    - LSTM layers for temporal dependencies
    - Fully connected classification head

    The CNN captures local patterns (like candlestick patterns),
    while the LSTM captures longer-term temporal relationships.
    """

    def __init__(
        self,
        n_features: int,
        cnn_channels: list[int] = None,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        """Initialize the CNN-LSTM model.

        Args:
            n_features: Number of input features
            cnn_channels: List of CNN channel sizes (default: [64, 128])
            kernel_size: CNN kernel size
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            n_classes: Number of output classes
        """
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128]

        self.lstm_hidden = lstm_hidden

        # CNN layers
        cnn_layers = []
        in_channels = n_features
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_length, n_features)

        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # CNN expects (batch, channels, seq_length)
        x = x.permute(0, 2, 1)

        # CNN forward
        x = self.cnn(x)  # (batch, cnn_channels[-1], seq_length')

        # Back to (batch, seq_length', channels)
        x = x.permute(0, 2, 1)

        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_length', lstm_hidden * 2)

        # Use last hidden state
        x = lstm_out[:, -1, :]  # (batch, lstm_hidden * 2)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class CNNLSTMClassifier(DeepRegimeClassifier):
    """CNN-LSTM hybrid regime classifier.

    Combines CNN for local pattern extraction with LSTM for temporal modeling.

    Usage:
        model = CNNLSTMClassifier(seq_length=60, hidden_size=128)
        model.fit(features_df, labels_series)
        predictions = model.predict(new_features)
    """

    def __init__(
        self,
        seq_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        cnn_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_classes: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str = None,
        random_state: int = 42,
    ):
        """Initialize the CNN-LSTM classifier.

        Args:
            seq_length: Number of time steps in input sequences
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            cnn_channels: List of CNN channel sizes
            kernel_size: CNN kernel size
            dropout: Dropout rate
            n_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device to use ('cuda', 'cpu', or None for auto)
            random_state: Random seed for reproducibility
        """
        super().__init__(
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_classes=n_classes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
            random_state=random_state,
        )
        self.cnn_channels = cnn_channels if cnn_channels is not None else [64, 128]
        self.kernel_size = kernel_size

    def _build_model(self) -> nn.Module:
        """Build the CNN-LSTM model.

        Returns:
            CNNLSTMModel instance
        """
        return CNNLSTMModel(
            n_features=self.n_features,
            cnn_channels=self.cnn_channels,
            kernel_size=self.kernel_size,
            lstm_hidden=self.hidden_size,
            lstm_layers=self.num_layers,
            dropout=self.dropout,
            n_classes=self.n_classes,
        )

    def save(self, path):
        """Save model with additional CNN-LSTM-specific config."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        import torch
        from pathlib import Path

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
                "cnn_channels": self.cnn_channels,
                "kernel_size": self.kernel_size,
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
