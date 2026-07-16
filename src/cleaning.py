import pandas as pd


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim leading/trailing whitespace and collapse multiple spaces.
    """
    df = df.copy()

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return df


def convert_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str]
) -> pd.DataFrame:

    df = df.copy()

    for col in numeric_columns:

        if col not in df.columns:
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def convert_date_columns(
    df: pd.DataFrame,
    date_columns: list[str]
) -> pd.DataFrame:

    df = df.copy()

    for col in date_columns:

        if col not in df.columns:
            continue

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    return df


def handle_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        median = df[col].median()

        if pd.notna(median):
            df[col] = df[col].fillna(median)

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:

        if not df[col].mode().empty:
            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    return df


def drop_duplicate_rows(
    df: pd.DataFrame
) -> pd.DataFrame:

    return (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


def clean_dataset(
    df: pd.DataFrame,
    numeric_columns: list[str],
    date_columns: list[str],
    trim_text: bool = True,
    convert_numeric: bool = True,
    convert_dates: bool = True,
    fill_missing: bool = True,
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:

    df = df.copy()

    report = {
        "text_trimmed": False,
        "numeric_columns_converted": 0,
        "date_columns_converted": 0,
        "missing_values_filled": 0,
        "duplicates_removed": 0,
    }

    if trim_text:
        df = clean_text(df)
        report["text_trimmed"] = True

    if convert_numeric:
        df = convert_numeric_columns(
            df,
            numeric_columns
        )
        report["numeric_columns_converted"] = len(
            numeric_columns
        )

    if convert_dates:
        df = convert_date_columns(
            df,
            date_columns
        )
        report["date_columns_converted"] = len(
            date_columns
        )

    if fill_missing:

        before = df.isna().sum().sum()

        df = handle_missing_values(df)

        after = df.isna().sum().sum()

        report["missing_values_filled"] = before - after

    if remove_duplicates:

        before = len(df)

        df = drop_duplicate_rows(df)

        after = len(df)

        report["duplicates_removed"] = before - after

    return df, report