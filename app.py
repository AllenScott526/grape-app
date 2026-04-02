from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from predict import predict_image

MODEL_PATH = "model_quant.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

app = Flask(__name__)

# Ensure static folder exists
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        disease, remedy = predict_image(filepath)

        return render_template('index.html',
                               prediction=disease,
                               remedy=remedy,
                               img_path=filepath)

    return "Error"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)