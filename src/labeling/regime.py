"""Regime labeling logic for training data."""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class RegimeType(str, Enum):
    """Market regime types."""

    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"


class RegimeLabeler:
    """Label market regimes based on price action and volatility.

    Regime classification logic:
    1. HIGH_VOL: When volatility is above the specified percentile
    2. BULL_TREND: Strong uptrend (return > threshold, not high vol)
    3. BEAR_TREND: Strong downtrend (return < -threshold, not high vol)
    4. SIDEWAYS: Everything else (range-bound, low volatility)

    Usage:
        labeler = RegimeLabeler(trend_threshold=0.02, vol_percentile=80)
        labels = labeler.label(ohlcv_df)
    """

    def __init__(
        self,
        trend_threshold: float = 0.02,
        vol_percentile: float = 80,
        return_window: int = 20,
        vol_window: int = 20,
        vol_lookback: int = 252,
        min_trend_days: int = 3,
    ):
        """Initialize the regime labeler.

        Args:
            trend_threshold: Minimum return to classify as trend (e.g., 0.02 = 2%)
            vol_percentile: Volatility percentile threshold for HIGH_VOL
            return_window: Window for calculating returns
            vol_window: Window for calculating volatility
            vol_lookback: Lookback period for volatility percentile
            min_trend_days: Minimum consecutive days to confirm trend
        """
        self.trend_threshold = trend_threshold
        self.vol_percentile = vol_percentile
        self.return_window = return_window
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.min_trend_days = min_trend_days

    def label(self, df: pd.DataFrame) -> pd.Series:
        """Label each row with a market regime.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Series with regime labels
        """
        close = df["close"]

        # Calculate indicators
        returns = close.pct_change(self.return_window)
        volatility = close.pct_change().rolling(self.vol_window).std()

        # Calculate volatility percentile
        vol_pct = volatility.rolling(self.vol_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50,
            raw=False
        )

        # Initialize labels
        labels = pd.Series(RegimeType.SIDEWAYS.value, index=df.index)

        # Label HIGH_VOL first (takes priority)
        high_vol_mask = vol_pct >= self.vol_percentile
        labels[high_vol_mask] = RegimeType.HIGH_VOL.value

        # Label trends (only if not high vol)
        not_high_vol = ~high_vol_mask

        bull_mask = (returns > self.trend_threshold) & not_high_vol
        bear_mask = (returns < -self.trend_threshold) & not_high_vol

        labels[bull_mask] = RegimeType.BULL_TREND.value
        labels[bear_mask] = RegimeType.BEAR_TREND.value

        # Apply minimum trend duration filter
        labels = self._apply_min_duration(labels)

        return labels

    def _apply_min_duration(self, labels: pd.Series) -> pd.Series:
        """Apply minimum duration filter to avoid regime flipping.

        Short regime periods are relabeled to the surrounding regime.
        """
        if self.min_trend_days <= 1:
            return labels

        result = labels.copy()

        # Find regime change points
        regime_changes = labels != labels.shift(1)
        change_indices = labels.index[regime_changes].tolist()

        # Add start and end
        if labels.index[0] not in change_indices:
            change_indices.insert(0, labels.index[0])
        change_indices.append(labels.index[-1])

        # Check each regime segment
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]

            # Get segment
            segment = labels.loc[start_idx:end_idx]

            # If segment is too short, relabel to previous regime
            if len(segment) < self.min_trend_days:
                if i > 0:
                    prev_regime = labels.loc[change_indices[i - 1]]
                    result.loc[start_idx:end_idx] = prev_regime

        return result

    def label_with_lookahead(
        self, df: pd.DataFrame, lookahead: int = 5
    ) -> pd.Series:
        """Label regimes using future information (for training only).

        This provides more accurate labels by looking ahead to confirm trends.
        Should NOT be used for live prediction.

        Args:
            df: DataFrame with OHLCV data
            lookahead: Number of periods to look ahead

        Returns:
            Series with regime labels
        """
        close = df["close"]

        # Forward returns
        fwd_returns = close.shift(-lookahead) / close - 1

        # Backward returns
        bwd_returns = close.pct_change(self.return_window)

        # Combined signal (both backward and forward)
        bull_signal = (bwd_returns > self.trend_threshold / 2) & (fwd_returns > 0)
        bear_signal = (bwd_returns < -self.trend_threshold / 2) & (fwd_returns < 0)

        # Volatility
        volatility = close.pct_change().rolling(self.vol_window).std()
        vol_pct = volatility.rolling(self.vol_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50,
            raw=False
        )

        # Labels
        labels = pd.Series(RegimeType.SIDEWAYS.value, index=df.index)

        high_vol_mask = vol_pct >= self.vol_percentile
        labels[high_vol_mask] = RegimeType.HIGH_VOL.value

        not_high_vol = ~high_vol_mask
        labels[bull_signal & not_high_vol] = RegimeType.BULL_TREND.value
        labels[bear_signal & not_high_vol] = RegimeType.BEAR_TREND.value

        return labels

    def get_regime_stats(self, labels: pd.Series) -> dict:
        """Get statistics about regime distribution.

        Args:
            labels: Series with regime labels

        Returns:
            Dictionary with regime statistics
        """
        counts = labels.value_counts()
        percentages = labels.value_counts(normalize=True) * 100

        stats = {
            "total_samples": len(labels),
            "regime_counts": counts.to_dict(),
            "regime_percentages": percentages.to_dict(),
        }

        # Calculate average regime duration
        durations = {}
        for regime in RegimeType:
            regime_mask = labels == regime.value
            if regime_mask.any():
                # Count consecutive regime periods
                changes = regime_mask != regime_mask.shift(1)
                groups = changes.cumsum()
                regime_groups = groups[regime_mask]
                if len(regime_groups) > 0:
                    group_lengths = regime_groups.groupby(regime_groups).size()
                    durations[regime.value] = group_lengths.mean()

        stats["avg_duration"] = durations

        return stats
