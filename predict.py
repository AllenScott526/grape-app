import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

remedies = {
    "Black Rot": "Remove infected leaves. Spray fungicide. Keep plant dry and clean.",
    "ESCA": "Prune infected parts. Avoid overwatering. Keep plant healthy and stress-free.",
    "Healthy": "Your plant is healthy 😊. Continue regular watering and sunlight.",
    "Leaf Blight": "Remove damaged leaves. Use fungicide spray. Avoid excess moisture."
}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if input_details[0]['dtype'] == np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = classes[np.argmax(prediction)]
    remedy = remedies[predicted_class]

    return predicted_class, remedy