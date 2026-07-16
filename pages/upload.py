import streamlit as st
import pandas as pd

from src.session import reset_pipeline
from src.data_loader import load_csv
from src.validator import validate_dataset


def show():
    st.title("📂 Upload Dataset")

    st.info("Upload your sales dataset here.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a sales dataset in CSV format."
    )

    if uploaded_file is not None:

        try:
            df = load_csv(uploaded_file)

        except ValueError as e:
            st.error(str(e))
            return

        # Validate 
        report = validate_dataset(df)

        if not report["valid"]:
            st.error("Dataset validation failed.")

            for error in report["errors"]:
                st.error(error)

            return

        # Save validated dataset
        st.session_state["raw_df"] = df
        st.session_state["dataset_name"] = uploaded_file.name
        st.session_state["validation_report"] = report

        reset_pipeline()

    if st.session_state["raw_df"] is not None:

        df = st.session_state["raw_df"]
        report = st.session_state["validation_report"]

        st.success("Dataset loaded successfully!")

        st.write(f"**Dataset:** {st.session_state['dataset_name']}")

        

        # Dataset Summary

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", report["info"]["rows"])

        with col2:
            st.metric("Columns", report["info"]["columns"])

        

        # Validation Report
        
        st.subheader("Validation Report")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Duplicate Rows",
                report["info"]["duplicate_rows"]
            )

        with col2:
            st.metric(
                "Columns with Missing Values",
                report["info"]["missing_count"]
            )

        with col3:
            st.metric("Validation", "Passed")

        # warnings
        for warning in report["warnings"]:
            st.warning(warning)

        if not report["warnings"]:
            st.success("No validation issues detected.")

        # Missing value details
        if report["info"]["missing_columns"]:

            st.subheader("Missing Values by Column")

            missing_df = (
                pd.DataFrame.from_dict(
                    report["info"]["missing_columns"],
                    orient="index",
                    columns=["Missing Values"]
                )
                .rename_axis("Column")
                .reset_index()
            )

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        # Dataset Preview

        st.subheader("Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )