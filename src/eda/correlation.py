# src/eda/correlation.py

import pandas as pd


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    correlation_matrix = numeric_df.corr(
        method="pearson",
        numeric_only=True
    )

    return correlation_matrix.round(3)