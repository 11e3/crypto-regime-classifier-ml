#!/usr/bin/env python
"""Analyze XGBoost model performance in detail."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import RegimeClassifier
from src.validation.walk_forward import walk_forward_validation


def main():
    # Load data
    print("=" * 70)
    print("XGBoost 단독 성능 분석")
    print("=" * 70)

    df = load_ohlcv("data/BTC.parquet")
    print(f"데이터: {len(df)} rows ({df.index.min().date()} ~ {df.index.max().date()})")

    # Extract features
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)

    # Generate labels
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    print(f"\n레이블 분포:")
    print(labels.value_counts())
    print(f"BULL_TREND 비율: {(labels == 'BULL_TREND').mean():.2%}")

    # Walk-forward validation
    print("\n" + "=" * 70)
    print("Walk-Forward Validation (37 Folds)")
    print("=" * 70)

    results_df = walk_forward_validation(
        features=features,
        labels=labels,
        model_class=RegimeClassifier,
        model_params={"model_type": "xgboost", "scale_features": True, "random_state": 42},
        train_size=500,
        test_size=60,
        step_size=60,
        verbose=False,
    )

    # Summary statistics
    print("\n" + "=" * 70)
    print("성능 요약")
    print("=" * 70)
    print(f"Fold 수: {len(results_df)}")
    print(f"Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Recall:    {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"F1:        {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")

    # Percentiles
    print("\n" + "-" * 40)
    print("Accuracy 분포:")
    print(f"  Min:  {results_df['accuracy'].min():.4f}")
    print(f"  25%:  {results_df['accuracy'].quantile(0.25):.4f}")
    print(f"  50%:  {results_df['accuracy'].quantile(0.50):.4f}")
    print(f"  75%:  {results_df['accuracy'].quantile(0.75):.4f}")
    print(f"  Max:  {results_df['accuracy'].max():.4f}")

    # Best and worst folds
    print("\n" + "=" * 70)
    print("최고/최저 성능 Fold")
    print("=" * 70)

    best_folds = results_df.nlargest(5, 'accuracy')[['fold', 'test_start', 'test_end', 'accuracy', 'f1']]
    worst_folds = results_df.nsmallest(5, 'accuracy')[['fold', 'test_start', 'test_end', 'accuracy', 'f1']]

    print("\n최고 성능 5개 Fold:")
    for _, row in best_folds.iterrows():
        print(f"  Fold {int(row['fold']):2d}: {row['test_start'].date()} ~ {row['test_end'].date()} | Acc: {row['accuracy']:.4f}, F1: {row['f1']:.4f}")

    print("\n최저 성능 5개 Fold:")
    for _, row in worst_folds.iterrows():
        print(f"  Fold {int(row['fold']):2d}: {row['test_start'].date()} ~ {row['test_end'].date()} | Acc: {row['accuracy']:.4f}, F1: {row['f1']:.4f}")

    # Accuracy by time period
    print("\n" + "=" * 70)
    print("기간별 성능 분석")
    print("=" * 70)

    results_df['year'] = pd.to_datetime(results_df['test_start']).dt.year
    yearly_perf = results_df.groupby('year').agg({
        'accuracy': ['mean', 'std', 'count'],
        'f1': 'mean'
    }).round(4)
    yearly_perf.columns = ['Accuracy Mean', 'Accuracy Std', 'Folds', 'F1 Mean']
    print("\n연도별 성능:")
    print(yearly_perf)

    # Analyze failure cases (F1 = 0)
    print("\n" + "=" * 70)
    print("실패 케이스 분석 (F1 = 0)")
    print("=" * 70)

    failed = results_df[results_df['f1'] == 0]
    print(f"\nF1=0 Fold 수: {len(failed)} / {len(results_df)} ({len(failed)/len(results_df):.1%})")

    if len(failed) > 0:
        print("\n실패한 Fold 상세:")
        for _, row in failed.iterrows():
            print(f"  Fold {int(row['fold']):2d}: {row['test_start'].date()} ~ {row['test_end'].date()} | Acc: {row['accuracy']:.4f}")

    # High performance analysis
    print("\n" + "=" * 70)
    print("고성능 구간 분석 (Accuracy >= 90%)")
    print("=" * 70)

    high_perf = results_df[results_df['accuracy'] >= 0.9]
    print(f"\n90%+ Fold 수: {len(high_perf)} / {len(results_df)} ({len(high_perf)/len(results_df):.1%})")
    print(f"평균 Accuracy: {high_perf['accuracy'].mean():.4f}")
    print(f"평균 F1: {high_perf['f1'].mean():.4f}")

    # Train final model and get feature importance
    print("\n" + "=" * 70)
    print("Feature Importance (전체 데이터 학습)")
    print("=" * 70)

    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    model = RegimeClassifier(model_type="xgboost", scale_features=True, random_state=42)
    model.fit(X, y, eval_split=0.2, verbose=False)

    importance = model.get_feature_importance()
    print("\nTop 15 Features:")
    for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
        print(f"  {i:2d}. {feat:<30} {imp:.4f}")

    # Consistency analysis
    print("\n" + "=" * 70)
    print("일관성 분석")
    print("=" * 70)

    # Rolling average of accuracy
    results_df['rolling_acc'] = results_df['accuracy'].rolling(window=5, min_periods=1).mean()

    print("\n5-Fold 이동평균 Accuracy:")
    print(f"  최소: {results_df['rolling_acc'].min():.4f}")
    print(f"  최대: {results_df['rolling_acc'].max():.4f}")
    print(f"  변동폭: {results_df['rolling_acc'].max() - results_df['rolling_acc'].min():.4f}")

    # Consecutive performance
    results_df['above_80'] = results_df['accuracy'] >= 0.8
    consecutive = results_df['above_80'].astype(int)
    max_consecutive = (consecutive * (consecutive.groupby((consecutive != consecutive.shift()).cumsum()).cumcount() + 1)).max()
    print(f"\n80%+ 연속 Fold 최대: {max_consecutive}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("결론 및 권장사항")
    print("=" * 70)

    print(f"""
XGBoost 단독 모델 분석 결과:

1. 전체 성능:
   - 평균 Accuracy: {results_df['accuracy'].mean():.2%}
   - 평균 F1 Score: {results_df['f1'].mean():.4f}

2. 안정성:
   - 표준편차: ±{results_df['accuracy'].std():.2%}
   - 90%+ 달성률: {len(high_perf)/len(results_df):.1%}
   - 실패율 (F1=0): {len(failed)/len(results_df):.1%}

3. 강점:
   - 대부분의 시장 상황에서 안정적 성능
   - Feature importance로 해석 가능
   - 학습/예측 속도 빠름

4. 약점:
   - 시장 전환기 예측 취약 (Fold 9, 27 등)
   - 일부 기간에서 한 클래스만 예측

5. 권장:
   - XGBoost 단독 사용 또는 RF와 50:50 앙상블
   - LSTM 제외 (성능 저하 우려)
   - 전환기 감지를 위한 추가 피처 고려
""")

    # Save results
    results_df.to_csv("xgb_walk_forward_detailed.csv", index=False)
    print("결과 저장: xgb_walk_forward_detailed.csv")


if __name__ == "__main__":
    main()
