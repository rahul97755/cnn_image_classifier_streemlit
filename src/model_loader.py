import streamlit as st
from tensorflow.keras.models import load_model as keras_load_model # type: ignore

@st.cache_resource
def load_trained_model():
    """Load and cache CNN model"""
    return keras_load_model("model/cifar10_cnn_augmented.h5") # type: ignore
