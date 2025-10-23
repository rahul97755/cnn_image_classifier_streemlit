import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model():
    """Load and cache CNN model"""
    return load_model("models/cifar10_cnn_augmented.h5")
