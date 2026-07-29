import streamlit as st


def show():
    st.set_page_config(page_title="Sales Forecasting")

    st.title("📈 Sales Forecasting")

    st.info(
        "Future sales forecasting will be available "
        "after model training."
    )