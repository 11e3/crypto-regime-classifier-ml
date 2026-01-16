"""Advanced feature extraction for deep learning models."""

import numpy as np
import pandas as pd


class AdvancedFeatures:
    """Extract advanced features optimized for deep learning regime classification.

    Features:
    - Multi-timeframe features (cross-timeframe signals)
    - Candlestick patterns
    - Momentum divergence
    - Fractal dimension / market complexity
    - Mean reversion indicators
    - Regime transition signals
    """

    def __init__(
        self,
        stoch_period: int = 14,
        williams_period: int = 14,
        cci_period: int = 20,
        mfi_period: int = 14,
    ):
        self.stoch_period = stoch_period
        self.williams_period = williams_period
        self.cci_period = cci_period
        self.mfi_period = mfi_period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all advanced features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with advanced features
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"]

        # 1. Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(high, low, close, self.stoch_period)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        features["stoch_crossover"] = (stoch_k - stoch_d).apply(np.sign)

        # 2. Williams %R
        features["williams_r"] = self._calculate_williams_r(high, low, close, self.williams_period)

        # 3. CCI (Commodity Channel Index)
        features["cci"] = self._calculate_cci(high, low, close, self.cci_period)

        # 4. MFI (Money Flow Index)
        features["mfi"] = self._calculate_mfi(high, low, close, volume, self.mfi_period)

        # 5. Candlestick patterns
        candle_features = self._extract_candlestick_patterns(open_, high, low, close)
        for name, values in candle_features.items():
            features[name] = values

        # 6. Price action features
        features["body_ratio"] = (close - open_).abs() / (high - low).replace(0, 1)
        features["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (high - low).replace(0, 1)
        features["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (high - low).replace(0, 1)

        # 7. Gap features
        features["gap"] = open_ / close.shift(1) - 1
        features["gap_filled"] = ((close >= close.shift(1)) & (features["gap"] < 0)) | \
                                 ((close <= close.shift(1)) & (features["gap"] > 0))
        features["gap_filled"] = features["gap_filled"].astype(float)

        # 8. Multi-timeframe momentum
        for period in [5, 10, 20, 40]:
            features[f"roc_{period}"] = (close / close.shift(period) - 1) * 100

        # 9. Momentum divergence (price vs volume)
        price_mom = close.pct_change(10)
        vol_mom = volume.pct_change(10)
        features["pv_divergence"] = price_mom - vol_mom
        features["pv_divergence_sign"] = np.sign(price_mom) != np.sign(vol_mom)
        features["pv_divergence_sign"] = features["pv_divergence_sign"].astype(float)

        # 10. Mean reversion indicators
        features["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        features["zscore_60"] = (close - close.rolling(60).mean()) / close.rolling(60).std()
        features["mean_reversion_signal"] = -features["zscore_20"]  # Negative z-score suggests buy

        # 11. Hurst exponent approximation (regime persistence)
        features["hurst_approx"] = self._calculate_hurst_approximation(close, 20)

        # 12. Efficiency ratio (trending vs noisy)
        features["efficiency_ratio"] = self._calculate_efficiency_ratio(close, 10)

        # 13. Keltner Channel position
        kc_upper, kc_middle, kc_lower = self._calculate_keltner_channel(high, low, close)
        features["kc_position"] = (close - kc_lower) / (kc_upper - kc_lower)

        # 14. Donchian Channel position
        dc_upper = high.rolling(20).max()
        dc_lower = low.rolling(20).min()
        features["dc_position"] = (close - dc_lower) / (dc_upper - dc_lower)

        # 15. Price acceleration
        velocity = close.pct_change()
        features["acceleration"] = velocity.diff()

        # 16. Consecutive up/down days
        features["consec_up"] = self._count_consecutive(close.pct_change() > 0)
        features["consec_down"] = self._count_consecutive(close.pct_change() < 0)

        # 17. Distance from EMA cloud
        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        features["ema_cloud_dist"] = (close - (ema_8 + ema_21) / 2) / close

        # 18. Volatility regime
        vol_short = close.pct_change().rolling(10).std()
        vol_long = close.pct_change().rolling(60).std()
        features["vol_regime"] = vol_short / vol_long

        # 19. Intraday range ratio
        daily_range = high - low
        features["range_expansion"] = daily_range / daily_range.rolling(20).mean()

        # 20. Normalized volume
        features["volume_zscore"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

        return features

    def _calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
        stoch_d = stoch_k.rolling(3).mean()

        return stoch_k, stoch_d

    def _calculate_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)
        return wr

    def _calculate_cci(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())

        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    def _calculate_mfi(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        money_flow_direction = np.sign(typical_price.diff())

        positive_flow = raw_money_flow.where(money_flow_direction > 0, 0)
        negative_flow = raw_money_flow.where(money_flow_direction < 0, 0)

        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()

        money_ratio = positive_mf / negative_mf.replace(0, 1)
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    def _extract_candlestick_patterns(
        self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict[str, pd.Series]:
        """Extract candlestick pattern features."""
        patterns = {}

        body = close - open_
        body_abs = body.abs()
        range_ = high - low
        upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

        # Doji (small body)
        patterns["doji"] = (body_abs / range_.replace(0, 1) < 0.1).astype(float)

        # Hammer (long lower shadow, small body at top)
        patterns["hammer"] = (
            (lower_shadow > 2 * body_abs) &
            (upper_shadow < body_abs) &
            (body > 0)
        ).astype(float)

        # Shooting star (long upper shadow, small body at bottom)
        patterns["shooting_star"] = (
            (upper_shadow > 2 * body_abs) &
            (lower_shadow < body_abs) &
            (body < 0)
        ).astype(float)

        # Engulfing patterns
        prev_body = body.shift(1)
        patterns["bullish_engulfing"] = (
            (prev_body < 0) &
            (body > 0) &
            (open_ < close.shift(1)) &
            (close > open_.shift(1))
        ).astype(float)

        patterns["bearish_engulfing"] = (
            (prev_body > 0) &
            (body < 0) &
            (open_ > close.shift(1)) &
            (close < open_.shift(1))
        ).astype(float)

        # Marubozu (no shadows)
        patterns["marubozu"] = (
            (upper_shadow / range_.replace(0, 1) < 0.05) &
            (lower_shadow / range_.replace(0, 1) < 0.05) &
            (body_abs / range_.replace(0, 1) > 0.9)
        ).astype(float)

        return patterns

    def _calculate_hurst_approximation(self, close: pd.Series, window: int) -> pd.Series:
        """Approximate Hurst exponent using R/S analysis.

        H > 0.5: Trending (persistent)
        H < 0.5: Mean-reverting (anti-persistent)
        H = 0.5: Random walk
        """
        def hurst_rs(prices):
            if len(prices) < 20:
                return 0.5

            returns = np.diff(np.log(prices))
            mean_return = np.mean(returns)
            cumulative_dev = np.cumsum(returns - mean_return)

            R = np.max(cumulative_dev) - np.min(cumulative_dev)
            S = np.std(returns)

            if S == 0:
                return 0.5

            RS = R / S
            n = len(returns)

            if RS <= 0 or n <= 1:
                return 0.5

            H = np.log(RS) / np.log(n)
            return np.clip(H, 0, 1)

        return close.rolling(window).apply(hurst_rs, raw=True)

    def _calculate_efficiency_ratio(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio.

        ER close to 1: Strong trend
        ER close to 0: Choppy/noisy market
        """
        change = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(period).sum()

        er = change / volatility.replace(0, 1)
        return er

    def _calculate_keltner_channel(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
        ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channel."""
        # Middle line is EMA
        middle = close.ewm(span=ema_period).mean()

        # ATR for channel width
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        upper = middle + multiplier * atr
        lower = middle - multiplier * atr

        return upper, middle, lower

    def _count_consecutive(self, condition: pd.Series) -> pd.Series:
        """Count consecutive True values."""
        groups = (~condition).cumsum()
        return condition.groupby(groups).cumsum()

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            # Stochastic
            "stoch_k", "stoch_d", "stoch_crossover",
            # Williams %R
            "williams_r",
            # CCI
            "cci",
            # MFI
            "mfi",
            # Candlestick patterns
            "doji", "hammer", "shooting_star",
            "bullish_engulfing", "bearish_engulfing", "marubozu",
            # Price action
            "body_ratio", "upper_shadow", "lower_shadow",
            # Gap
            "gap", "gap_filled",
            # ROC multi-timeframe
            "roc_5", "roc_10", "roc_20", "roc_40",
            # Divergence
            "pv_divergence", "pv_divergence_sign",
            # Mean reversion
            "zscore_20", "zscore_60", "mean_reversion_signal",
            # Complexity
            "hurst_approx", "efficiency_ratio",
            # Channels
            "kc_position", "dc_position",
            # Momentum
            "acceleration", "consec_up", "consec_down",
            # EMA cloud
            "ema_cloud_dist",
            # Volatility
            "vol_regime", "range_expansion", "volume_zscore",
        ]
