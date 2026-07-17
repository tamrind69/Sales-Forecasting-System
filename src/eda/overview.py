import pandas as pd


def generate_overview(
    df: pd.DataFrame,
    numeric_columns: list,
    categorical_columns: list,
    date_columns: list,
) -> dict:
    """
    Generate a high-level overview of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    numeric_columns : list
        User-selected numerical columns.
    categorical_columns : list
        User-selected categorical columns.
    date_columns : list
        User-selected date columns.

    Returns
    -------
    dict
        Dictionary containing dataset summary information.
    """

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)

    column_summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values,
    })

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_usage_mb": round(memory_usage, 2),
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "datetime_columns": len(date_columns),
        "column_summary": column_summary,
    }