import streamlit as st

DEFAULT_SESSION_STATE = {
    "raw_df": None,
    "processed_df": None,
    "dataset_name": None,
    "model": None,
    "metrics": None,
    "predictions": None,
}

def initialize_session():
    """Initialize application session state."""

    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline():
    """Reset everything derived from the uploaded dataset."""

    st.session_state["processed_df"] = None
    st.session_state["model"] = None
    st.session_state["metrics"] = None
    st.session_state["predictions"] = None