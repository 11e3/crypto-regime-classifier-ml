"""Volume-based feature extraction."""

import numpy as np
import pandas as pd


class VolumeFeatures:
    """Extract volume-based features for regime classification.

    Features:
    - Volume ratio (vs MA)
    - OBV trend
    - Volume momentum
    - Volume-price divergence
    """

    def __init__(
        self,
        ma_periods: list[int] = None,
        obv_ma_period: int = 20,
    ):
        self.ma_periods = ma_periods or [5, 10, 20]
        self.obv_ma_period = obv_ma_period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all volume features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)

        volume = df["volume"]
        close = df["close"]

        # Volume ratios vs moving averages
        for period in self.ma_periods:
            vol_ma = volume.rolling(period).mean()
            features[f"volume_ratio_{period}"] = volume / vol_ma

        # Volume momentum
        features["volume_change"] = volume.pct_change()
        features["volume_ma_ratio"] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # On-Balance Volume (OBV)
        obv = self._calculate_obv(close, volume)
        features["obv"] = obv
        features["obv_ma"] = obv.rolling(self.obv_ma_period).mean()
        features["obv_trend"] = (obv - features["obv_ma"]) / features["obv_ma"].abs().replace(0, 1)

        # OBV slope (trend direction)
        features["obv_slope"] = obv.diff(5) / 5

        # Volume-weighted price indicators
        features["vwap_ratio"] = self._calculate_vwap_ratio(df)

        # Volume-price divergence
        price_change = close.pct_change(5)
        volume_change = volume.pct_change(5)
        features["volume_price_corr"] = (
            price_change.rolling(20).corr(volume_change)
        )

        # Accumulation/Distribution
        features["ad_line"] = self._calculate_ad_line(df)
        features["ad_trend"] = features["ad_line"].diff(10)

        # Volume volatility
        features["volume_volatility"] = volume.rolling(20).std() / volume.rolling(20).mean()

        return features

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
        return obv

    def _calculate_vwap_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate VWAP ratio (current price vs VWAP)."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (
            (typical_price * df["volume"]).rolling(period).sum()
            / df["volume"].rolling(period).sum()
        )
        return df["close"] / vwap

    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)

        # Money Flow Volume
        mfv = mfm * volume

        # A/D Line (cumulative)
        ad = mfv.cumsum()
        return ad

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        names = []
        for period in self.ma_periods:
            names.append(f"volume_ratio_{period}")
        names.extend([
            "volume_change", "volume_ma_ratio",
            "obv", "obv_ma", "obv_trend", "obv_slope",
            "vwap_ratio",
            "volume_price_corr",
            "ad_line", "ad_trend",
            "volume_volatility",
        ])
        return names
