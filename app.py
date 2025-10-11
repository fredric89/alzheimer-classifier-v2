import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# ------------------------------
# LOAD ONNX MODEL
# ------------------------------
session = ort.InferenceSession("mobilenetv2_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Class labels
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def softmax(x):
    """Apply softmax to convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

# ------------------------------
# STREAMLIT UI SETUP
# ------------------------------
st.set_page_config(
    page_title="Alzheimer's Stage Classifier with MobilenetV2",
    layout="centered"
)
st.title(" Alzheimer's Stage Classifier using MobilenetV2 model")
st.write("Upload a brain MRI image to predict the Alzheimer's stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # ------------------------------
        # PREPROCESS IMAGE (RGB)
        # ------------------------------
        img = image.convert("RGB")  # Ensure 3 channels
        img = img.resize((224, 224))  # Resize to model input size

        # Convert to numpy array and normalize like training
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalize

        # Transpose to [C,H,W] and add batch dimension [1,3,224,224]
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        st.write("Input shape for ONNX model:", img_array.shape)

        # ------------------------------
        # RUN INFERENCE
        # ------------------------------
        pred = session.run([output_name], {input_name: img_array})[0]
        
        # Apply softmax to convert logits to probabilities
        probabilities = softmax(pred[0])
        
        predicted_class = labels[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        # ------------------------------
        # DISPLAY RESULTS
        # ------------------------------
        st.success(f" Prediction: {predicted_class} ({confidence:.2f}%)")

        st.subheader("Class Probabilities:")
        for label, p in zip(labels, probabilities):
            percentage = p * 100
            st.write(f"{label}: {percentage:.2f}%")
            # Ensure progress bar value is between 0-100
            progress_value = max(0, min(100, int(percentage)))
            st.progress(progress_value)

    except Exception as e:
        st.error(f" Inference failed: {e}")
