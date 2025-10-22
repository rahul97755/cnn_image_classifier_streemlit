import streamlit as st
from PIL import Image
from src.model_loader import load_model
from src.preprocessing import preprocess_image
from src.predict import predict

# --- Config ---
st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠")

# --- Load model ---
model_path = "models/cifar10_cnn_augmented.h5"
model = load_model(model_path)

# --- UI ---
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image — model will predict among 10 categories.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img_array = preprocess_image(image)
    predicted_class, confidence = predict(model, img_array)

    st.markdown(f"### 🏷️ Prediction: **{predicted_class.upper()}**")
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2f}")
