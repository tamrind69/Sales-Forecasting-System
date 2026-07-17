import pandas as pd


def analyze_categorical(
    df: pd.DataFrame,
    categorical_columns: list,
) -> dict:
    """
    Analyze categorical columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    categorical_columns : list
        User-selected categorical columns.

    Returns
    -------
    dict
        Categorical summary statistics.
    """

    if not categorical_columns:
        return {
            "categorical_columns": [],
            "summary": pd.DataFrame(),
        }

    categorical_df = df[categorical_columns]

    summary = pd.DataFrame({
        "Column": categorical_df.columns,
        "Unique Values": categorical_df.nunique().values,
        "Most Frequent": categorical_df.mode().iloc[0].values,
        "Frequency": categorical_df.apply(
            lambda col: col.value_counts().iloc[0]
        ).values,
    })

    return {
        "categorical_columns": categorical_columns,
        "summary": summary,
    }