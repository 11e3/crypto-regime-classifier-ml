"""Price-based feature extraction."""

import numpy as np
import pandas as pd


class PriceFeatures:
    """Extract price-based features for regime classification.

    Features:
    - Returns (1d, 5d, 20d)
    - Volatility (rolling std)
    - RSI
    - MACD
    - Bollinger Band position
    """

    def __init__(
        self,
        return_periods: list[int] = None,
        volatility_window: int = 20,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_window: int = 20,
        bb_std: float = 2.0,
    ):
        self.return_periods = return_periods or [1, 5, 20]
        self.volatility_window = volatility_window
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_window = bb_window
        self.bb_std = bb_std

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all price features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Returns
        for period in self.return_periods:
            features[f"return_{period}d"] = close.pct_change(period)

        # Volatility (rolling standard deviation of returns)
        daily_returns = close.pct_change()
        features["volatility"] = daily_returns.rolling(self.volatility_window).std()
        features["volatility_zscore"] = (
            features["volatility"] - features["volatility"].rolling(60).mean()
        ) / features["volatility"].rolling(60).std()

        # RSI
        features["rsi"] = self._calculate_rsi(close, self.rsi_period)

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        # Bollinger Bands position
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            close, self.bb_window, self.bb_std
        )
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle

        # Price momentum
        features["momentum_10"] = close / close.shift(10) - 1
        features["momentum_20"] = close / close.shift(20) - 1

        # True Range and ATR
        tr = self._calculate_true_range(high, low, close)
        features["atr"] = tr.rolling(14).mean()
        features["atr_pct"] = features["atr"] / close

        return features

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, close: pd.Series, fast: int, slow: int, signal: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(
        self, close: pd.Series, window: int, num_std: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    def _calculate_true_range(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Calculate True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        names = []
        for period in self.return_periods:
            names.append(f"return_{period}d")
        names.extend([
            "volatility", "volatility_zscore",
            "rsi",
            "macd", "macd_signal", "macd_histogram",
            "bb_position", "bb_width",
            "momentum_10", "momentum_20",
            "atr", "atr_pct",
        ])
        return names
