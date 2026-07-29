import pandas as pd


# Monthly Sales Preparation

def prepare_monthly_sales(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
):
    """
    Aggregate transaction-level sales into monthly sales.
    """

    data = df[
        [
            date_column,
            target_column,
        ]
    ].copy()

    # Ensure date column is datetime

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    # Remove rows with invalid dates or missing target values

    data = data.dropna(
        subset=[
            date_column,
            target_column,
        ]
    )

    # Sort chronologically

    data = data.sort_values(
        by=date_column
    )

    # Aggregate target values by month

    monthly_sales = (
        data
        .set_index(date_column)
        .resample("MS")[target_column]
        .sum()
        .reset_index()
    )

    return monthly_sales


# Forecast Feature Generation

def create_forecast_features(
    monthly_sales: pd.DataFrame,
    date_column: str,
    target_column: str,
):
    """
    Create historical and seasonal features
    for monthly sales forecasting.
    """

    data = monthly_sales.copy()

    # Recent sales history

    data["Lag_1"] = (
        data[target_column]
        .shift(1)
    )

    data["Lag_2"] = (
        data[target_column]
        .shift(2)
    )

    data["Lag_3"] = (
        data[target_column]
        .shift(3)
    )

    # Same month previous year

    data["Lag_12"] = (
        data[target_column]
        .shift(12)
    )

    # Calendar features

    data["Month"] = (
        data[date_column]
        .dt.month
    )

    data["Quarter"] = (
        data[date_column]
        .dt.quarter
    )

    # Lag_12 requires 12 months of history

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    return data


# Chronological Train-Test Split

def split_time_series(
    forecast_data: pd.DataFrame,
    date_column: str,
    target_column: str,
    test_size: float = 0.2,
):
    """
    Split forecasting data chronologically into
    training and testing sets.
    """

    feature_columns = [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_12",
        "Month",
        "Quarter",
    ]

    split_index = int(
        len(forecast_data) * (1 - test_size)
    )

    train_data = forecast_data.iloc[
        :split_index
    ]

    test_data = forecast_data.iloc[
        split_index:
    ]

    X_train = train_data[
        feature_columns
    ].copy()

    y_train = train_data[
        target_column
    ].copy()

    X_test = test_data[
        feature_columns
    ].copy()

    y_test = test_data[
        target_column
    ].copy()

    train_dates = train_data[
        date_column
    ].copy()

    test_dates = test_data[
        date_column
    ].copy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "feature_columns": feature_columns,
    }