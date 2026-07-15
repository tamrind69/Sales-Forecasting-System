import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting System")

st.markdown("""
Welcome to the **Sales Forecasting System**.

This application helps you:

- 📂 Upload sales datasets
- 📊 Perform Exploratory Data Analysis (EDA)
- 🤖 Train Machine Learning models
- 📈 Forecast future sales
- 📄 Generate downloadable reports

Use the sidebar to navigate through the application.
""")