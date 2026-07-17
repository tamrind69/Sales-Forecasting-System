import pandas as pd


def generate_overview(df: pd.DataFrame) -> dict:
    """
    Generate a high-level overview of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

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
        "Unique Values": df.nunique().values
    })

    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_usage_mb": round(memory_usage, 2),
        "numeric_columns": len(df.select_dtypes(include="number").columns),
        "categorical_columns": len(
            df.select_dtypes(include=["object", "category"]).columns
        ),
        "datetime_columns": len(
            df.select_dtypes(include=["datetime", "datetimetz"]).columns
        ),
        "column_summary": column_summary
    }

    return overview