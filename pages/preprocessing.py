import streamlit as st
import src.cleaning as cleaning


def show():

    st.title("🧹 Data Cleaning & Preprocessing")

    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state.raw_df

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Column Configuration")

    date_columns = st.multiselect(
        "Date Columns",
        options=df.columns.tolist(),
        default=[]
    )

    numeric_columns = st.multiselect(
        "Numeric Columns",
        options=df.columns.tolist(),
        default=df.select_dtypes(include="number").columns.tolist()
    )

    st.divider()

    st.subheader("Cleaning Options")

    trim_text = st.checkbox(
        "Trim whitespace",
        value=True
    )

    convert_numeric = st.checkbox(
        "Convert numeric columns",
        value=True
    )

    convert_dates = st.checkbox(
        "Convert date column",
        value=True
    )

    fill_missing = st.checkbox(
        "Fill missing values",
        value=True
    )

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

    st.divider()

    if st.button(
        "Run Cleaning",
        type="primary"
    ):

        cleaned_df, report = cleaning.clean_dataset(
            df=df,
            numeric_columns=numeric_columns,
            date_columns=date_columns,
            trim_text=trim_text,
            convert_numeric=convert_numeric,
            convert_dates=convert_dates,
            fill_missing=fill_missing,
            remove_duplicates=remove_duplicates,
        )

        st.session_state.processed_df = cleaned_df

        # Save user-defined column types
        st.session_state.numeric_columns = numeric_columns
        st.session_state.date_columns = date_columns

        st.session_state.categorical_columns = [
            col
            for col in cleaned_df.columns
            if col not in numeric_columns
            and col not in date_columns
        ]

        st.success("Cleaning completed successfully!")

        st.subheader("Cleaning Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Numeric Columns Converted",
                report["numeric_columns_converted"]
            )

            st.metric(
                "Missing Values Filled",
                report["missing_values_filled"]
            )

        with col2:
            st.metric(
                "Date Columns Converted",
                report["date_columns_converted"]
            )

            st.metric(
                "Duplicate Rows Removed",
                report["duplicates_removed"]
            )

    st.divider()

    st.subheader("Preprocessing")

    st.info(
        "Feature encoding, scaling, train-test split, and other preprocessing "
        "steps will be implemented in the next stage of the project."
    )