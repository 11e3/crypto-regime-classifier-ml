"""Regime classifier model."""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class RegimeClassifier:
    """Market regime classifier.

    Supports multiple model types:
    - random_forest: Random Forest Classifier
    - gradient_boosting: Gradient Boosting Classifier
    - xgboost: XGBoost Classifier
    - logistic: Logistic Regression

    Usage:
        model = RegimeClassifier(model_type="random_forest")
        model.fit(features, labels)
        predictions = model.predict(new_features)
    """

    SUPPORTED_MODELS = ["random_forest", "gradient_boosting", "xgboost", "logistic"]

    def __init__(
        self,
        model_type: str = "random_forest",
        scale_features: bool = True,
        random_state: int = 42,
        **model_params,
    ):
        """Initialize the regime classifier.

        Args:
            model_type: Type of model to use
            scale_features: Whether to scale features before training
            random_state: Random seed for reproducibility
            **model_params: Additional parameters passed to the underlying model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.scale_features = scale_features
        self.random_state = random_state
        self.model_params = model_params

        self.model = self._create_model()
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoder = LabelEncoder() if model_type == "xgboost" else None
        self.feature_names: Optional[list[str]] = None
        self.is_fitted = False

    def _create_model(self):
        """Create the underlying sklearn model."""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)

        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": self.random_state,
            }
            default_params.update(self.model_params)
            return GradientBoostingClassifier(**default_params)

        elif self.model_type == "xgboost":
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": -1,
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
            }
            default_params.update(self.model_params)
            return XGBClassifier(**default_params)

        elif self.model_type == "logistic":
            default_params = {
                "max_iter": 1000,
                "random_state": self.random_state,
            }
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_split: float = 0.2,
        verbose: bool = True,
    ) -> "RegimeClassifier":
        """Train the classifier.

        Args:
            X: Feature DataFrame
            y: Labels Series
            eval_split: Fraction of data to use for evaluation
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        # Store feature names
        self.feature_names = list(X.columns)

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=eval_split, random_state=self.random_state, stratify=y
        )

        # Scale features
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values

        # Train model
        if verbose:
            print(f"Training {self.model_type} classifier...")
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Encode labels for XGBoost
        if self.label_encoder:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val)
            self.model.fit(X_train_scaled, y_train_encoded)
        else:
            self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True

        # Evaluate
        if verbose:
            train_acc = self.model.score(X_train_scaled, y_train_encoded if self.label_encoder else y_train)
            val_acc = self.model.score(X_val_scaled, y_val_encoded if self.label_encoder else y_val)
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")

            y_pred_raw = self.model.predict(X_val_scaled)
            y_pred = self.label_encoder.inverse_transform(y_pred_raw) if self.label_encoder else y_pred_raw
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred))

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

        # Validate features
        self._validate_features(X)

        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X[self.feature_names])
        else:
            X_scaled = X[self.feature_names].values

        predictions_raw = self.model.predict(X_scaled)

        # Decode predictions for XGBoost
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions_raw)
        else:
            predictions = predictions_raw

        if len(predictions) == 1:
            return predictions[0]

        return pd.Series(predictions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> dict:
        """Predict probability for each regime.

        Args:
            X: Feature DataFrame (single row expected)

        Returns:
            Dictionary with probability for each regime
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_features(X)

        if self.scaler:
            X_scaled = self.scaler.transform(X[self.feature_names])
        else:
            X_scaled = X[self.feature_names].values

        probas = self.model.predict_proba(X_scaled)

        # Get class labels (decode for XGBoost)
        if self.label_encoder:
            classes = self.label_encoder.classes_
        else:
            classes = self.model.classes_

        if len(probas) == 1:
            return dict(zip(classes, probas[0]))

        # Return DataFrame for multiple predictions
        return pd.DataFrame(probas, index=X.index, columns=classes)

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

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores.

        Returns:
            Series with feature names and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # For logistic regression, use mean absolute coefficients
            importance = np.abs(self.model.coef_).mean(axis=0)
        else:
            raise ValueError(f"Model type {self.model_type} does not support feature importance")

        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict:
        """Perform cross-validation.

        Args:
            X: Feature DataFrame
            y: Labels Series
            cv: Number of cross-validation folds

        Returns:
            Dictionary with CV scores
        """
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")

        return {
            "cv_scores": scores,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
        }

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
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "scale_features": self.scale_features,
            "random_state": self.random_state,
        }

        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RegimeClassifier":
        """Load model from file.

        Args:
            path: Path to the saved model

        Returns:
            Loaded RegimeClassifier instance
        """
        model_data = joblib.load(path)

        instance = cls(
            model_type=model_data["model_type"],
            scale_features=model_data["scale_features"],
            random_state=model_data["random_state"],
            **model_data["model_params"],
        )

        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.label_encoder = model_data.get("label_encoder")
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = True

        return instance
