import joblib
import streamlit as st

from src.config import MODEL_PATH


@st.cache_resource
def load_model_package():
    """Load the trained model package once per Streamlit session."""
    package = joblib.load(MODEL_PATH)
    return package["model"], package["scaler"], package["features"]
