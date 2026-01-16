"""Transformer-based regime classifier."""

import math

import torch
import torch.nn as nn

from .base import DeepRegimeClassifier


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer.

    Adds positional information to the input embeddings using
    sinusoidal functions.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_length, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder model for sequence classification.

    Architecture:
    - Linear projection to model dimension
    - Positional encoding
    - Transformer encoder layers
    - Global average pooling
    - Classification head
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        n_classes: int = 3,
        max_len: int = 500,
    ):
        """Initialize the Transformer model.

        Args:
            n_features: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            n_classes: Number of output classes
            max_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_length, n_features)

        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # Project to model dimension
        x = self.input_projection(x)  # (batch, seq_length, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_length, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TransformerClassifier(DeepRegimeClassifier):
    """Transformer-based regime classifier.

    Uses self-attention to capture long-range dependencies in time series data.

    Usage:
        model = TransformerClassifier(seq_length=60, hidden_size=128)
        model.fit(features_df, labels_series)
        predictions = model.predict(new_features)
    """

    def __init__(
        self,
        seq_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.2,
        n_classes: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str = None,
        random_state: int = 42,
    ):
        """Initialize the Transformer classifier.

        Args:
            seq_length: Number of time steps in input sequences
            hidden_size: Model dimension (d_model)
            num_layers: Number of transformer encoder layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
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
        self.n_heads = n_heads
        self.d_ff = d_ff

    def _build_model(self) -> nn.Module:
        """Build the Transformer model.

        Returns:
            TransformerModel instance
        """
        return TransformerModel(
            n_features=self.n_features,
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            n_classes=self.n_classes,
            max_len=self.seq_length + 100,
        )

    def save(self, path):
        """Save model with additional Transformer-specific config."""
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
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
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
