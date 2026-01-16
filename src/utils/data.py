"""Data loading utilities."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_ohlcv(
    path: Union[str, Path],
    date_column: str = "timestamp",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data from CSV file.

    Args:
        path: Path to CSV file
        date_column: Name of the date/timestamp column
        parse_dates: Whether to parse dates

    Returns:
        DataFrame with OHLCV data
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Read CSV
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Handle common date column names
    date_cols = ["timestamp", "date", "time", "datetime", "index"]
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col and parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.sort_index()

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_multiple_symbols(
    data_dir: Union[str, Path],
    symbols: list[str] = None,
    pattern: str = "*.csv",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple symbols.

    Args:
        data_dir: Directory containing data files
        symbols: List of symbols to load (file names without extension)
        pattern: Glob pattern for finding files

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data = {}

    if symbols:
        files = [data_dir / f"{symbol}.csv" for symbol in symbols]
    else:
        files = list(data_dir.glob(pattern))

    for file_path in files:
        if file_path.exists():
            symbol = file_path.stem
            try:
                df = load_ohlcv(file_path)
                data[symbol] = df
                print(f"Loaded {symbol}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {symbol}: {e}")

    return data


def prepare_training_data(
    df: pd.DataFrame,
    feature_extractor,
    labeler,
    test_size: float = 0.2,
    use_lookahead: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare training and test data.

    Args:
        df: OHLCV DataFrame
        feature_extractor: FeatureExtractor instance
        labeler: RegimeLabeler instance
        test_size: Fraction of data for testing
        use_lookahead: Whether to use lookahead labeling

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Extract features
    features = feature_extractor.transform(df)

    # Generate labels
    if use_lookahead:
        labels = labeler.label_with_lookahead(df)
    else:
        labels = labeler.label(df)

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Time-based split (no shuffling for time series)
    split_idx = int(len(features) * (1 - test_size))

    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_test = labels.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str = "1D",
) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe.

    Args:
        df: OHLCV DataFrame with datetime index
        timeframe: Target timeframe (e.g., '1H', '4H', '1D', '1W')

    Returns:
        Resampled DataFrame
    """
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def add_symbol_prefix(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add symbol prefix to column names.

    Useful when combining multiple symbols.

    Args:
        df: DataFrame
        symbol: Symbol name to prefix

    Returns:
        DataFrame with prefixed columns
    """
    df = df.copy()
    df.columns = [f"{symbol}_{col}" for col in df.columns]
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate OHLCV data quality.

    Args:
        df: OHLCV DataFrame

    Returns:
        Dictionary with validation results
    """
    results = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "date_range": (df.index.min(), df.index.max()) if hasattr(df.index, "min") else None,
        "issues": [],
    }

    # Check for OHLC consistency
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if invalid_ohlc.any():
        results["issues"].append(f"Invalid OHLC: {invalid_ohlc.sum()} rows")

    # Check for negative prices
    negative = (df[["open", "high", "low", "close"]] < 0).any(axis=1)
    if negative.any():
        results["issues"].append(f"Negative prices: {negative.sum()} rows")

    # Check for zero volume
    zero_vol = df["volume"] == 0
    if zero_vol.any():
        results["issues"].append(f"Zero volume: {zero_vol.sum()} rows")

    # Check for duplicates
    if hasattr(df.index, "duplicated"):
        dupes = df.index.duplicated()
        if dupes.any():
            results["issues"].append(f"Duplicate dates: {dupes.sum()} rows")

    return results
