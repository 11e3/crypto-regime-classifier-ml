"""LSTM-based regime classifier."""

import torch
import torch.nn as nn

from .base import DeepRegimeClassifier


class LSTMModel(nn.Module):
    """Bidirectional LSTM model for sequence classification.

    Architecture:
    - Bidirectional LSTM layers with dropout
    - Optional attention mechanism
    - Fully connected classification head
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
        use_attention: bool = True,
    ):
        """Initialize the LSTM model.

        Args:
            n_features: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            n_classes: Number of output classes
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_length, n_features)

        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_length, hidden_size * 2)

        if self.use_attention:
            # Attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch, seq_length, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size * 2)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]  # (batch, hidden_size * 2)

        # Classification
        out = self.dropout(context)
        out = self.fc(out)

        return out


class LSTMClassifier(DeepRegimeClassifier):
    """LSTM-based regime classifier.

    A bidirectional LSTM with optional attention for market regime classification.

    Usage:
        model = LSTMClassifier(seq_length=60, hidden_size=128)
        model.fit(features_df, labels_series)
        predictions = model.predict(new_features)
    """

    def __init__(
        self,
        seq_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
        use_attention: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str = None,
        random_state: int = 42,
    ):
        """Initialize the LSTM classifier.

        Args:
            seq_length: Number of time steps in input sequences
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            n_classes: Number of output classes
            use_attention: Whether to use attention mechanism
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
        self.use_attention = use_attention

    def _build_model(self) -> nn.Module:
        """Build the LSTM model.

        Returns:
            LSTMModel instance
        """
        return LSTMModel(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_classes=self.n_classes,
            use_attention=self.use_attention,
        )

    def save(self, path):
        """Save model with additional LSTM-specific config."""
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
                "dropout": self.dropout,
                "n_classes": self.n_classes,
                "use_attention": self.use_attention,
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
