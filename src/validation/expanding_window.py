"""Expanding window validation for unsupervised models."""

from typing import Optional, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def expanding_window_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    model_class: Any,
    model_params: dict,
    initial_train_size: int = 300,
    test_size: int = 30,
    step_size: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """Perform expanding window validation for unsupervised models.

    Unlike walk-forward, expanding window keeps all historical data:
    - Start with initial training window
    - Test on next window
    - Expand training to include test window
    - Repeat

    This is ideal for unsupervised models that benefit from more data
    to learn the underlying distribution.

    Args:
        features: Feature DataFrame
        labels: Labels Series (for evaluation only, not training)
        model_class: Unsupervised model class
        model_params: Parameters for the model
        initial_train_size: Initial training window size
        test_size: Test window size
        step_size: How many steps to move forward each iteration
        verbose: Whether to print progress

    Returns:
        DataFrame with results for each fold
    """
    results = []
    n_samples = len(features)

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    fold = 0
    train_end = initial_train_size

    while train_end + test_size <= n_samples:
        fold += 1
        test_start = train_end
        test_end = train_end + test_size

        # Expanding window: train on all data from start to train_end
        train_features = features.iloc[:train_end]
        test_features = features.iloc[test_start:test_end]
        test_labels = labels.iloc[test_start:test_end]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold}")
            print(f"Train: {train_features.index.min().date()} to {train_features.index.max().date()} ({len(train_features)} samples)")
            print(f"Test:  {test_features.index.min().date()} to {test_features.index.max().date()} ({len(test_features)} samples)")

        # Train model (unsupervised - no labels)
        model = model_class(**model_params)
        model.fit(train_features, verbose=False)

        # Predict
        predictions = model.predict(test_features)

        if isinstance(predictions, str):
            predictions = pd.Series([predictions], index=[test_features.index[-1]])

        # Evaluate against true labels
        common_idx = predictions.index.intersection(test_labels.index)
        if len(common_idx) == 0:
            print(f"  Skipping fold {fold} - no matching indices")
            train_end += step_size
            continue

        y_true = test_labels.loc[common_idx]
        y_pred = predictions.loc[common_idx]

        # Calculate metrics - use weighted average to handle label mismatches
        # Unsupervised models may predict different labels than the true labels
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results.append({
            "fold": fold,
            "train_start": train_features.index.min(),
            "train_end": train_features.index.max(),
            "test_start": test_features.index.min(),
            "test_end": test_features.index.max(),
            "train_samples": len(train_features),
            "test_samples": len(y_true),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        if verbose:
            print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # Expand window
        train_end += step_size

    return pd.DataFrame(results)


def evaluate_unsupervised_model(
    model_class: Any,
    model_params: dict,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str = "Model",
    initial_train_size: int = 300,
    test_size: int = 30,
    step_size: int = 30,
    verbose: bool = True,
) -> dict:
    """Evaluate an unsupervised model using expanding window validation.

    Args:
        model_class: Unsupervised model class
        model_params: Model parameters
        features: Feature DataFrame
        labels: True labels for evaluation
        model_name: Name for display
        initial_train_size: Initial training window
        test_size: Test window size
        step_size: Step size
        verbose: Print progress

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"{model_name} Expanding Window Validation")
        print("=" * 60)

    results_df = expanding_window_validation(
        features=features,
        labels=labels,
        model_class=model_class,
        model_params=model_params,
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
        verbose=verbose,
    )

    if len(results_df) == 0:
        return {
            "model_name": model_name,
            "n_folds": 0,
            "mean_accuracy": np.nan,
            "std_accuracy": np.nan,
            "mean_f1": np.nan,
            "std_f1": np.nan,
            "results_df": results_df,
        }

    # Summary
    summary = {
        "model_name": model_name,
        "n_folds": len(results_df),
        "mean_accuracy": results_df["accuracy"].mean(),
        "std_accuracy": results_df["accuracy"].std(),
        "mean_precision": results_df["precision"].mean(),
        "std_precision": results_df["precision"].std(),
        "mean_recall": results_df["recall"].mean(),
        "std_recall": results_df["recall"].std(),
        "mean_f1": results_df["f1"].mean(),
        "std_f1": results_df["f1"].std(),
        "results_df": results_df,
    }

    if verbose:
        print(f"\n{model_name} Summary")
        print("-" * 40)
        print(f"Number of folds: {summary['n_folds']}")
        print(f"Accuracy:  {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
        print(f"Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
        print(f"F1:        {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    return summary
