import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model(model_path: str):
    """Load and cache the trained CNN model."""
    model = tf.keras.models.load_model(model_path)
    return model
