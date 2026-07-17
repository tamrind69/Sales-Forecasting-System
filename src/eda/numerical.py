import pandas as pd


def analyze_numerical(
    df: pd.DataFrame,
    numeric_columns: list,
) -> dict:
    """
    Analyze numerical columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    numeric_columns : list
        User-selected numerical columns.

    Returns
    -------
    dict
        Numerical summary statistics.
    """

    if not numeric_columns:
        return {
            "numeric_columns": [],
            "summary": pd.DataFrame(),
        }

    numeric_df = df[numeric_columns]

    summary = pd.DataFrame({
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Std": numeric_df.std(),
        "Min": numeric_df.min(),
        "Q1": numeric_df.quantile(0.25),
        "Q3": numeric_df.quantile(0.75),
        "Max": numeric_df.max(),
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurt(),
    }).round(2)

    summary.index.name = "Column"
    summary.reset_index(inplace=True)

    return {
        "numeric_columns": numeric_columns,
        "summary": summary,
    }