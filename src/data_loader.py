import pandas as pd


COMMON_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]


def load_csv(uploaded_file):
    """
    Load a CSV file by trying common encodings.
    """

    for encoding in COMMON_ENCODINGS:

        try:
            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Unable to read the CSV. Unsupported encoding."
    )