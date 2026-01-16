"""Training utilities for deep learning models."""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


class Trainer:
    """Trainer for deep learning regime classifiers.

    Handles the training loop with:
    - Early stopping
    - Learning rate scheduling
    - Class-weighted loss
    - Progress logging
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            class_weights: Optional class weights for imbalanced data
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate

        # Setup loss function with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Setup optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ) -> list[dict]:
        """Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            List of training history dictionaries
        """
        history = []
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        if verbose:
            print(f"Training on {self.device}")
            print(f"Training samples: {len(train_loader.dataset)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            print("-" * 60)

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc, train_f1 = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc, val_f1 = self._validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Record history
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model = self.model.to(self.device)

        if verbose:
            print("-" * 60)
            print(f"Best validation loss: {best_val_loss:.4f}")

        return history

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float, float]:
        """Run one training epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for sequences, labels in train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Record metrics
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, f1

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float, float]:
        """Run one validation epoch.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * len(labels)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, f1


class EarlyStopping:
    """Early stopping handler.

    Monitors a metric and stops training when it stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
