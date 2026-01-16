"""Main feature extractor combining all feature modules."""

import pandas as pd

from src.features.price import PriceFeatures
from src.features.volume import VolumeFeatures
from src.features.structure import StructureFeatures


class FeatureExtractor:
    """Main feature extractor that combines price, volume, and structure features.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.transform(ohlcv_df)
    """

    def __init__(
        self,
        include_price: bool = True,
        include_volume: bool = True,
        include_structure: bool = True,
        price_params: dict = None,
        volume_params: dict = None,
        structure_params: dict = None,
    ):
        self.include_price = include_price
        self.include_volume = include_volume
        self.include_structure = include_structure

        self.price_extractor = PriceFeatures(**(price_params or {})) if include_price else None
        self.volume_extractor = VolumeFeatures(**(volume_params or {})) if include_volume else None
        self.structure_extractor = StructureFeatures(**(structure_params or {})) if include_structure else None

    def transform(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Extract all features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            dropna: Whether to drop rows with NaN values

        Returns:
            DataFrame with all extracted features
        """
        # Validate input columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        features_list = []

        if self.price_extractor:
            price_features = self.price_extractor.transform(df)
            features_list.append(price_features)

        if self.volume_extractor:
            volume_features = self.volume_extractor.transform(df)
            features_list.append(volume_features)

        if self.structure_extractor:
            structure_features = self.structure_extractor.transform(df)
            features_list.append(structure_features)

        if not features_list:
            raise ValueError("At least one feature type must be enabled")

        # Combine all features
        features = pd.concat(features_list, axis=1)

        # Handle duplicate columns (e.g., 'atr' appears in both price and structure)
        features = features.loc[:, ~features.columns.duplicated()]

        if dropna:
            features = features.dropna()

        return features

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names."""
        names = []

        if self.price_extractor:
            names.extend(self.price_extractor.get_feature_names())

        if self.volume_extractor:
            names.extend(self.volume_extractor.get_feature_names())

        if self.structure_extractor:
            names.extend(self.structure_extractor.get_feature_names())

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        return unique_names

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Fit the extractor (for API compatibility, currently no-op)."""
        return self

    def fit_transform(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, dropna=dropna)
