# src/eda/outliers.py

import pandas as pd


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in numerical columns using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Outlier summary for each numerical column.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    summary = []

    for column in numeric_df.columns:

        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (
            (numeric_df[column] < lower_bound)
            | (numeric_df[column] > upper_bound)
        )

        outlier_count = outlier_mask.sum()

        summary.append({
            "Column": column,
            "Outliers": outlier_count,
            "Percentage": round(
                (outlier_count / len(numeric_df)) * 100,
                2
            )
        })

    return pd.DataFrame(summary)