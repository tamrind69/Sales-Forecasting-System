import streamlit as st
from src.session import initialize_session

initialize_session()

from pages import (
    home,
    upload,
    preprocessing,
    eda,
    modeling,
    forecasting,
    report,
)

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
)

# Navigation

pg = st.navigation(
    [
        st.Page(home.show, title="Home", icon="🏠",url_path="home", default=True),
        st.Page(upload.show, title="Upload Dataset",url_path="upload", icon="📂"),
        st.Page(preprocessing.show, title="Data Preprocessing",url_path="preprocessing", icon="🧹"),
        st.Page(eda.show, title="Exploratory Data Analysis", url_path="eda", icon="📊"),
        st.Page(modeling.show, title="Model Training", url_path="modeling", icon="🤖"),
        st.Page(forecasting.show, title="Sales Forecasting", url_path="forecasting", icon="📈"),
        st.Page(report.show, title="Reports", url_path="report", icon="📄"),
    ]
)

pg.run()