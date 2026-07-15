import streamlit as st
import pandas as pd

from src.session import reset_pipeline
from src.data_loader import load_csv


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

        st.session_state["raw_df"] = df
        st.session_state["dataset_name"] = uploaded_file.name

        reset_pipeline()
    
    if st.session_state["raw_df"] is not None:

        df = st.session_state["raw_df"]

        st.success("Dataset loaded successfully!")

        st.write(f"**Dataset:** {st.session_state['dataset_name']}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        st.subheader("Preview")
        st.dataframe(df.head())