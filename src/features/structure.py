"""Market structure feature extraction."""

import numpy as np
import pandas as pd


class StructureFeatures:
    """Extract market structure features for regime classification.

    Features:
    - Higher highs/lows count
    - MA alignment (5, 20, 60)
    - ATR percentile
    - Support/resistance proximity
    - Trend strength
    """

    def __init__(
        self,
        ma_periods: list[int] = None,
        lookback_hh_hl: int = 20,
        atr_period: int = 14,
        atr_percentile_window: int = 252,
    ):
        self.ma_periods = ma_periods or [5, 20, 60]
        self.lookback_hh_hl = lookback_hh_hl
        self.atr_period = atr_period
        self.atr_percentile_window = atr_percentile_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all market structure features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with structure features
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Moving averages and alignment
        mas = {}
        for period in self.ma_periods:
            mas[period] = close.rolling(period).mean()
            features[f"ma_{period}"] = mas[period]
            features[f"close_to_ma_{period}"] = (close - mas[period]) / mas[period]

        # MA alignment score (-1 to 1: bearish to bullish)
        features["ma_alignment"] = self._calculate_ma_alignment(mas)

        # MA slopes
        for period in self.ma_periods:
            features[f"ma_{period}_slope"] = mas[period].diff(5) / mas[period].shift(5)

        # Higher highs / Lower lows count
        hh_count, hl_count, lh_count, ll_count = self._count_pivots(high, low, self.lookback_hh_hl)
        features["higher_highs_count"] = hh_count
        features["higher_lows_count"] = hl_count
        features["lower_highs_count"] = lh_count
        features["lower_lows_count"] = ll_count

        # Trend score based on pivots
        features["pivot_trend_score"] = (
            (features["higher_highs_count"] + features["higher_lows_count"])
            - (features["lower_highs_count"] + features["lower_lows_count"])
        ) / self.lookback_hh_hl

        # ATR percentile
        atr = self._calculate_atr(high, low, close, self.atr_period)
        features["atr"] = atr
        features["atr_percentile"] = atr.rolling(self.atr_percentile_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Distance from recent high/low (support/resistance)
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        features["dist_from_high"] = (close - rolling_high) / rolling_high
        features["dist_from_low"] = (close - rolling_low) / rolling_low

        # Price position within range
        features["range_position"] = (close - rolling_low) / (rolling_high - rolling_low)

        # Trend strength (ADX-like)
        features["trend_strength"] = self._calculate_trend_strength(high, low, close)

        # Consolidation detection
        features["consolidation"] = self._detect_consolidation(high, low, close)

        # Breakout potential
        features["breakout_potential"] = self._calculate_breakout_potential(
            close, rolling_high, rolling_low, atr
        )

        return features

    def _calculate_ma_alignment(self, mas: dict[int, pd.Series]) -> pd.Series:
        """Calculate MA alignment score.

        Returns score from -1 (fully bearish) to 1 (fully bullish).
        Bullish: MA5 > MA20 > MA60
        Bearish: MA5 < MA20 < MA60
        """
        periods = sorted(self.ma_periods)

        # Count bullish alignments
        score = pd.Series(0.0, index=mas[periods[0]].index)

        for i in range(len(periods) - 1):
            short_ma = mas[periods[i]]
            long_ma = mas[periods[i + 1]]
            score += np.where(short_ma > long_ma, 1, -1)

        # Normalize to -1 to 1
        max_score = len(periods) - 1
        return score / max_score

    def _count_pivots(
        self, high: pd.Series, low: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Count higher highs, higher lows, lower highs, lower lows."""

        def count_hh(window):
            highs = window.values
            count = 0
            for i in range(1, len(highs)):
                if highs[i] > highs[i - 1]:
                    count += 1
            return count

        def count_ll(window):
            lows = window.values
            count = 0
            for i in range(1, len(lows)):
                if lows[i] < lows[i - 1]:
                    count += 1
            return count

        def count_lh(window):
            highs = window.values
            count = 0
            for i in range(1, len(highs)):
                if highs[i] < highs[i - 1]:
                    count += 1
            return count

        def count_hl(window):
            lows = window.values
            count = 0
            for i in range(1, len(lows)):
                if lows[i] > lows[i - 1]:
                    count += 1
            return count

        hh_count = high.rolling(lookback).apply(count_hh, raw=False)
        ll_count = low.rolling(lookback).apply(count_ll, raw=False)
        lh_count = high.rolling(lookback).apply(count_lh, raw=False)
        hl_count = low.rolling(lookback).apply(count_hl, raw=False)

        return hh_count, hl_count, lh_count, ll_count

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _calculate_trend_strength(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate trend strength (simplified ADX-like indicator)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = self._calculate_atr(high, low, close, period)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(period).mean()

        return adx / 100  # Normalize to 0-1

    def _detect_consolidation(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> pd.Series:
        """Detect consolidation (range-bound market).

        Returns value between 0 (trending) and 1 (consolidating).
        """
        # Range as percentage of price
        range_pct = (high.rolling(window).max() - low.rolling(window).min()) / close

        # Normalize: smaller range = more consolidation
        range_percentile = range_pct.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Invert so 1 = consolidation, 0 = trending
        return 1 - range_percentile

    def _calculate_breakout_potential(
        self,
        close: pd.Series,
        rolling_high: pd.Series,
        rolling_low: pd.Series,
        atr: pd.Series,
    ) -> pd.Series:
        """Calculate breakout potential score.

        Higher values indicate price is close to breaking out of range.
        """
        dist_to_high = (rolling_high - close) / atr
        dist_to_low = (close - rolling_low) / atr

        # Closer to either boundary = higher breakout potential
        min_dist = pd.concat([dist_to_high, dist_to_low], axis=1).min(axis=1)

        # Normalize: 0 = at boundary, 1 = far from boundary
        # Invert so higher = closer to breakout
        return 1 / (1 + min_dist)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        names = []

        for period in self.ma_periods:
            names.extend([f"ma_{period}", f"close_to_ma_{period}", f"ma_{period}_slope"])

        names.extend([
            "ma_alignment",
            "higher_highs_count", "higher_lows_count",
            "lower_highs_count", "lower_lows_count",
            "pivot_trend_score",
            "atr", "atr_percentile",
            "dist_from_high", "dist_from_low", "range_position",
            "trend_strength", "consolidation", "breakout_potential",
        ])
        return names
