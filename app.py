import streamlit as st
from PIL import Image
from src.model_loader import load_trained_model
from src.preprocessing import preprocess_image
from src.predict import predict

st.set_page_config(page_title="🧠 CIFAR-10 CNN Classifier", layout="centered")
st.title("🧠 CIFAR-10 CNN Image Classifier")
st.markdown("Upload an image (32x32) and classify it into one of 10 categories using a trained CNN model.")

model = load_trained_model()

uploaded_file = st.file_uploader("📸 Upload Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔄 Predicting...")
    processed = preprocess_image(image)
    label, confidence = predict(model, processed)

    st.success(f"✅ Prediction: **{label}** ({confidence:.2f}% confidence)")
