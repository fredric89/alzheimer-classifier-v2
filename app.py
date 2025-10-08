import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Class labels (adjust if different)
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Title
st.title("🧠 Alzheimer’s Stage Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (example — adjust to match your model’s training)
    img = image.resize((128, 128))       # Example size — change if needed
    img_array = np.array(img) / 255.0    # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: img_array})[0]

    # Get prediction
    predicted_class = labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"✅ Prediction: {predicted_class} ({confidence:.2f}%)")

    # Show confidence per class
    st.subheader("Class probabilities:")
    for label, p in zip(labels, pred[0]):
        st.write(f"{label}: {p*100:.2f}%")
