import tensorflow as tf
import numpy as np
import json

# Load artifacts
model = tf.keras.models.load_model("plant_disease_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

def predict_leaf(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = 100 * np.max(predictions[0])
    
    return predicted_class, confidence

# Test on a single leaf image
label, conf = predict_leaf("data/test_leaf.jpg")
print(f"Prediction: {label} ({conf:.2f}% confidence)")