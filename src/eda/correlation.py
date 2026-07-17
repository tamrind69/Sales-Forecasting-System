import pandas as pd


def compute_correlation(
    df: pd.DataFrame,
    numeric_columns: list,
) -> pd.DataFrame:
    """
    Compute the correlation matrix for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    numeric_columns : list
        User-selected numerical columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """

    if len(numeric_columns) < 2:
        return pd.DataFrame()

    numeric_df = df[numeric_columns]

    correlation_matrix = numeric_df.corr().round(2)

    return correlation_matrix