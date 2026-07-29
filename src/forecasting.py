import pandas as pd


def prepare_monthly_sales(
    df,
    date_column,
    target_column,
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

    # Remove invalid dates and missing target values

    data = data.dropna(
        subset=[
            date_column,
            target_column,
        ]
    )

    # Aggregate sales by month

    monthly_sales = (
        data
        .set_index(date_column)
        .resample("MS")[target_column]
        .sum()
        .reset_index()
    )

    return monthly_sales

def create_lag_features(
    monthly_sales,
    target_column,
    n_lags=3,
):
    """
    Create lag features from monthly sales.
    """

    data = monthly_sales.copy()

    for lag in range(1, n_lags + 1):
        data[f"Lag_{lag}"] = data[target_column].shift(lag)

    # Remove rows without sufficient historical data

    data = data.dropna().reset_index(drop=True)

    return data

def split_time_series(
    lagged_data,
    date_column,
    target_column,
    test_size=0.2,
):
    """
    Split lagged time-series data chronologically
    into training and testing sets.
    """

    feature_columns = [
        col
        for col in lagged_data.columns
        if col.startswith("Lag_")
    ]

    split_index = int(
        len(lagged_data) * (1 - test_size)
    )

    train_data = lagged_data.iloc[:split_index]
    test_data = lagged_data.iloc[split_index:]

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    train_dates = train_data[date_column]
    test_dates = test_data[date_column]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "feature_columns": feature_columns,
    }