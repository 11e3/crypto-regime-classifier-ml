#!/usr/bin/env python
"""Test minimal feature sets."""

import pandas as pd
from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import RegimeClassifier
from src.validation.walk_forward import walk_forward_validation


FEATURE_SETS = {
    "full_51": None,  # All features
    "reduced_18": [
        "return_1d", "return_5d", "return_20d",
        "volatility", "atr_pct",
        "rsi", "momentum_20", "macd_histogram",
        "volume_ratio_20", "obv_trend", "volume_price_corr",
        "ma_alignment", "ma_20_slope", "trend_strength", "pivot_trend_score",
        "bb_position", "range_position", "breakout_potential",
    ],
    "minimal_10": [
        "return_20d",      # 수익률 (momentum_20과 동일)
        "volatility",      # 변동성
        "rsi",             # 모멘텀
        "macd_histogram",  # 추세 방향
        "volume_ratio_20", # 거래량
        "ma_alignment",    # 이평선 정렬
        "ma_20_slope",     # 추세 기울기
        "bb_position",     # 가격 위치
        "trend_strength",  # 추세 강도
        "breakout_potential", # 돌파 가능성
    ],
    "core_7": [
        "return_20d",      # 20일 수익률
        "volatility",      # 변동성
        "rsi",             # RSI
        "macd_histogram",  # MACD
        "volume_ratio_20", # 거래량 비율
        "ma_alignment",    # MA 정렬
        "trend_strength",  # 추세 강도
    ],
    "ultra_5": [
        "return_20d",      # 모멘텀
        "volatility",      # 변동성
        "rsi",             # 과매수/과매도
        "ma_alignment",    # 추세 방향
        "volume_ratio_20", # 거래량
    ],
}


def main():
    print("=" * 70)
    print("Feature Set Comparison")
    print("=" * 70)

    # Load data
    df = load_ohlcv("data/BTC.parquet")
    extractor = FeatureExtractor(include_advanced=False)
    features_full = extractor.transform(df)

    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    common_idx = features_full.index.intersection(labels.index)
    y = labels.loc[common_idx]

    results = []

    for name, feature_list in FEATURE_SETS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print("=" * 50)

        if feature_list is None:
            X = features_full.loc[common_idx]
        else:
            X = features_full.loc[common_idx, feature_list]

        print(f"Features: {len(X.columns)}")

        wf_results = walk_forward_validation(
            features=X,
            labels=y,
            model_class=RegimeClassifier,
            model_params={"model_type": "xgboost", "scale_features": True, "random_state": 42},
            train_size=500,
            test_size=60,
            step_size=60,
            verbose=False,
        )

        acc_mean = wf_results['accuracy'].mean()
        acc_std = wf_results['accuracy'].std()
        f1_mean = wf_results['f1'].mean()
        f1_std = wf_results['f1'].std()

        results.append({
            "name": name,
            "n_features": len(X.columns),
            "accuracy": acc_mean,
            "acc_std": acc_std,
            "f1": f1_mean,
            "f1_std": f1_std,
        })

        print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"F1:       {f1_mean:.4f} ± {f1_std:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Name':<15} {'Features':<10} {'Accuracy':<20} {'F1':<20}")
    print("-" * 65)
    for r in results:
        acc = f"{r['accuracy']:.4f} ± {r['acc_std']:.4f}"
        f1 = f"{r['f1']:.4f} ± {r['f1_std']:.4f}"
        print(f"{r['name']:<15} {r['n_features']:<10} {acc:<20} {f1:<20}")


if __name__ == "__main__":
    main()
