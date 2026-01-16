"""Walk-forward validation for supervised models."""

from typing import Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def walk_forward_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    model_class: Any,
    model_params: dict,
    train_size: int = 500,
    test_size: int = 60,
    step_size: int = 60,
    is_deep_learning: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Perform walk-forward validation for supervised models.

    Walk-forward validation simulates real-world trading:
    - Train on historical data
    - Test on next period
    - Roll window forward
    - Repeat

    This prevents look-ahead bias in supervised learning.

    Args:
        features: Feature DataFrame
        labels: Labels Series
        model_class: Supervised model class
        model_params: Parameters for the model
        train_size: Training window size
        test_size: Test window size
        step_size: How many steps to move forward each iteration
        is_deep_learning: Whether model is deep learning (needs seq_length handling)
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
    start_idx = 0

    while start_idx + train_size + test_size <= n_samples:
        fold += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size

        # Split data (rolling window)
        train_features = features.iloc[start_idx:train_end]
        train_labels = labels.iloc[start_idx:train_end]
        test_features = features.iloc[train_end:test_end]
        test_labels = labels.iloc[train_end:test_end]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold}")
            print(f"Train: {train_features.index.min().date()} to {train_features.index.max().date()} ({len(train_features)} samples)")
            print(f"Test:  {test_features.index.min().date()} to {test_features.index.max().date()} ({len(test_features)} samples)")

        # Train model
        model = model_class(**model_params)

        # Handle different model interfaces
        if hasattr(model, 'fit'):
            if 'eval_split' in model.fit.__code__.co_varnames:
                model.fit(train_features, train_labels, eval_split=0.15, verbose=False)
            else:
                model.fit(train_features, train_labels, verbose=False)

        # Predict
        predictions = model.predict(test_features)

        if isinstance(predictions, str):
            predictions = pd.Series([predictions], index=[test_features.index[-1]])

        # Handle deep learning seq_length offset
        if is_deep_learning:
            seq_length = model_params.get("seq_length", 60)
            valid_test_labels = test_labels.iloc[seq_length - 1:]
        else:
            valid_test_labels = test_labels

        if len(predictions) == 0 or len(valid_test_labels) == 0:
            print(f"  Skipping fold {fold} - insufficient samples")
            start_idx += step_size
            continue

        # Align predictions with labels
        common_idx = predictions.index.intersection(valid_test_labels.index)
        if len(common_idx) == 0:
            print(f"  Skipping fold {fold} - no matching indices")
            start_idx += step_size
            continue

        y_true = valid_test_labels.loc[common_idx]
        y_pred = predictions.loc[common_idx]

        # Calculate metrics
        unique_labels = sorted(labels.unique())
        acc = accuracy_score(y_true, y_pred)

        if len(unique_labels) == 2:
            prec = precision_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        else:
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

        # Move forward
        start_idx += step_size

    return pd.DataFrame(results)


def evaluate_supervised_model(
    model_class: Any,
    model_params: dict,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str = "Model",
    train_size: int = 500,
    test_size: int = 60,
    step_size: int = 60,
    is_deep_learning: bool = False,
    verbose: bool = True,
) -> dict:
    """Evaluate a supervised model using walk-forward validation.

    Args:
        model_class: Supervised model class
        model_params: Model parameters
        features: Feature DataFrame
        labels: True labels
        model_name: Name for display
        train_size: Training window
        test_size: Test window
        step_size: Step size
        is_deep_learning: Whether model is deep learning
        verbose: Print progress

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"{model_name} Walk-Forward Validation")
        print("=" * 60)

    results_df = walk_forward_validation(
        features=features,
        labels=labels,
        model_class=model_class,
        model_params=model_params,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        is_deep_learning=is_deep_learning,
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
