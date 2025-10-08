import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import onnxruntime as ort

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

LABELS = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    image = Image.open(file).convert('RGB')
    input_data = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    probs = outputs[0][0]

    exp_probs = np.exp(probs - np.max(probs))
    softmax = exp_probs / np.sum(exp_probs)

    pred_idx = np.argmax(softmax)
    prediction = LABELS[pred_idx]
    confidence = float(softmax[pred_idx]) * 100

    all_probs = {LABELS[i]: float(softmax[i]) * 100 for i in range(len(LABELS))}

    return render_template('index.html', prediction=f"{prediction} — {confidence:.2f}%", probs=all_probs)

if __name__ == "__main__":
    app.run()
