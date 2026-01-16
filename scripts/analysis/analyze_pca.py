#!/usr/bin/env python
"""Analyze PCA for feature reduction.

Usage:
    python analyze_pca.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv


def main():
    # Load data
    print("=" * 60)
    print("PCA Analysis for Feature Reduction")
    print("=" * 60)

    df = load_ohlcv("data/BTC.parquet")
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)

    print(f"Original features: {len(features.columns)}")
    print(f"Samples: {len(features)}")

    # Handle missing values
    features_clean = features.dropna()
    print(f"After dropna: {len(features_clean)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_clean)

    # Full PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Explained variance
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    print("\n" + "=" * 60)
    print("Cumulative Explained Variance")
    print("=" * 60)
    targets = [0.80, 0.85, 0.90, 0.95, 0.99]
    for target in targets:
        n_components = np.argmax(cumulative_var >= target) + 1
        print(f"  {target:.0%} variance: {n_components} components")

    # Detailed breakdown
    print("\n" + "-" * 40)
    print("Top 20 Components:")
    print("-" * 40)
    for i in range(min(20, len(pca_full.explained_variance_ratio_))):
        var = pca_full.explained_variance_ratio_[i]
        cum = cumulative_var[i]
        print(f"  PC{i+1:2d}: {var:6.2%} (cumulative: {cum:6.2%})")

    # Feature importance per component
    print("\n" + "=" * 60)
    print("Top Features per Principal Component")
    print("=" * 60)

    feature_names = features_clean.columns.tolist()
    for i in range(min(5, len(pca_full.components_))):
        print(f"\nPC{i+1} ({pca_full.explained_variance_ratio_[i]:.2%} variance):")
        loadings = pd.Series(pca_full.components_[i], index=feature_names)
        top_pos = loadings.nlargest(3)
        top_neg = loadings.nsmallest(3)
        print("  Positive loadings:")
        for feat, val in top_pos.items():
            print(f"    {feat}: {val:.3f}")
        print("  Negative loadings:")
        for feat, val in top_neg.items():
            print(f"    {feat}: {val:.3f}")

    # Correlation-based feature selection (alternative)
    print("\n" + "=" * 60)
    print("High Correlation Feature Groups")
    print("=" * 60)

    corr_matrix = features_clean.corr().abs()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append({
                    "feat1": corr_matrix.columns[i],
                    "feat2": corr_matrix.columns[j],
                    "corr": corr_matrix.iloc[i, j],
                })

    if high_corr_pairs:
        print(f"\nPairs with correlation > 0.9: {len(high_corr_pairs)}")
        for pair in sorted(high_corr_pairs, key=lambda x: -x["corr"])[:15]:
            print(f"  {pair['feat1']:<25} - {pair['feat2']:<25}: {pair['corr']:.3f}")

    # Suggest reduced feature set
    print("\n" + "=" * 60)
    print("Recommended Feature Reduction")
    print("=" * 60)

    # Group highly correlated features and pick representative
    redundant_features = set()
    for pair in high_corr_pairs:
        # Keep the first, mark second as redundant
        redundant_features.add(pair["feat2"])

    essential_features = [f for f in feature_names if f not in redundant_features]
    print(f"\nAfter removing redundant features: {len(essential_features)}")
    print(f"Removed: {len(redundant_features)}")

    # Manual curation based on domain knowledge
    curated_features = [
        # Returns (keep 3)
        "return_1d", "return_5d", "return_20d",
        # Volatility (keep 2)
        "volatility", "atr_pct",
        # Momentum (keep 3)
        "rsi", "momentum_20", "macd_histogram",
        # Volume (keep 3)
        "volume_ratio_20", "obv_trend", "volume_price_corr",
        # Trend (keep 4)
        "ma_alignment", "ma_20_slope", "trend_strength", "pivot_trend_score",
        # Bollinger (keep 1)
        "bb_position",
        # Range (keep 2)
        "range_position", "breakout_potential",
    ]

    print(f"\nCurated feature set: {len(curated_features)} features")
    print("-" * 40)
    for f in curated_features:
        print(f"  - {f}")

    # Save plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, 21), pca_full.explained_variance_ratio_[:20])
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Variance by Component")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, "b-")
    plt.axhline(y=0.95, color="r", linestyle="--", label="95%")
    plt.axhline(y=0.90, color="g", linestyle="--", label="90%")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Variance")
    plt.legend()

    plt.tight_layout()
    plt.savefig("pca_analysis.png", dpi=150)
    print(f"\nPlot saved: pca_analysis.png")


if __name__ == "__main__":
    main()
