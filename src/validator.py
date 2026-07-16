import pandas as pd


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the uploaded dataset.

    Returns
    -------
    dict
        {
            "valid": bool,
            "errors": list,
            "warnings": list,
            "info": dict
        }
    """

    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }

    
    # Basic Info
    
    report["info"]["rows"] = df.shape[0]
    report["info"]["columns"] = df.shape[1]

    
    # Empty 
    
    if df.empty:
        report["valid"] = False
        report["errors"].append("The uploaded dataset is empty.")
        return report

    
    # Duplicate 
    
    duplicate_rows = df.duplicated().sum()

    report["info"]["duplicate_rows"] = duplicate_rows

    if duplicate_rows > 0:
        report["warnings"].append(
            f"{duplicate_rows} duplicate row(s) found."
        )

    
    # Missing 
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    report["info"]["missing_count"] = len(missing)
    report["info"]["missing_columns"] = missing.to_dict()

    if not missing.empty:
        report["warnings"].append(
            f"{len(missing)} column(s) contain missing values."
        )

    
    # Duplicate Columns
    
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()

    report["info"]["duplicate_columns"] = duplicate_columns

    if duplicate_columns:
        report["valid"] = False
        report["errors"].append(
            f"Duplicate column names found: {', '.join(duplicate_columns)}"
        )

    
    # Blank Column 
    
    blank_columns = [
        col
        for col in df.columns
        if str(col).strip() == ""
    ]

    report["info"]["blank_columns"] = blank_columns

    if blank_columns:
        report["valid"] = False
        report["errors"].append(
            "Dataset contains blank column names."
        )

    return report