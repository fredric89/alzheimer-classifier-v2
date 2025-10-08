import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# ------------------------------
# LOAD ONNX MODEL
# ------------------------------
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Class labels
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(
    page_title="Alzheimer's Stage Classifier",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Alzheimer's Stage Classifier")
st.write("Upload a brain MRI image to predict the Alzheimer's stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    try:
        # ------------------------------
        # PREPROCESS IMAGE (Grayscale)
        # ------------------------------
        img = image.convert("L")  # Convert to grayscale
        img = img.resize((224, 224))

        # Convert to numpy and normalize (same as training)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalize

        # Add channel and batch dimensions: [B,C,H,W]
        img_array = np.expand_dims(img_array, axis=0)  # [C,H,W]
        img_array = np.expand_dims(img_array, axis=0)  # [B,C,H,W]

        # Debug: print shape and dtype
        st.write("Input shape for ONNX:", img_array.shape)
        st.write("Input dtype:", img_array.dtype)

        # ------------------------------
        # RUN INFERENCE
        # ------------------------------
        pred = session.run([output_name], {input_name: img_array})[0]
        predicted_class = labels[np.argmax(pred)]
        confidence = np.max(pred) * 100

        # Show prediction
        st.success(f"✅ Prediction: {predicted_class} ({confidence:.2f}%)")

        # Show probabilities for all classes
        st.subheader("Class probabilities:")
        for label, p in zip(labels, pred[0]):
            st.progress(p)  # Optional: progress bar for each class
            st.write(f"{label}: {p*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Inference failed: {e}")
