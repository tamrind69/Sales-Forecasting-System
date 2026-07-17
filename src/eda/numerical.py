import pandas as pd


def analyze_numerical(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Numerical analysis results.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "numeric_columns": [],
            "summary": pd.DataFrame()
        }

    summary = pd.DataFrame({
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Std": numeric_df.std(),
        "Min": numeric_df.min(),
        "Q1": numeric_df.quantile(0.25),
        "Q3": numeric_df.quantile(0.75),
        "Max": numeric_df.max(),
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurt()
    })

    summary = summary.round(3)
    summary.index.name = "Column"
    summary.reset_index(inplace=True)

    return {
        "numeric_columns": numeric_df.columns.tolist(),
        "summary": summary
    }