import pandas as pd


def detect_outliers(
    df: pd.DataFrame,
    numeric_columns: list,
) -> pd.DataFrame:
    """
    Detect outliers in numerical columns using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    numeric_columns : list
        User-selected numerical columns.

    Returns
    -------
    pd.DataFrame
        Summary of outliers for each numerical column.
    """

    if not numeric_columns:
        return pd.DataFrame()

    numeric_df = df[numeric_columns]

    outlier_summary = []

    for column in numeric_df.columns:

        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (numeric_df[column] < lower_bound)
            | (numeric_df[column] > upper_bound)
        )

        outlier_summary.append({
            "Column": column,
            "Outliers": outliers.sum(),
            "Percentage": round(
                outliers.mean() * 100,
                2
            )
        })

    return pd.DataFrame(outlier_summary)