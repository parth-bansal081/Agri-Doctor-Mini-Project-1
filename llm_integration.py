import os
import sys
import json
import numpy as np
import tensorflow as tf
from google import genai
from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("API KEY NOT FOUND")
    sys.exit(1)

model = tf.keras.models.load_model("plant_disease_model.keras")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

def predict_disease(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    raw_logits = model.predict(img_batch, verbose=0)
    probabilities = tf.nn.softmax(raw_logits[0]).numpy()
    top_class_idx = np.argmax(probabilities)
    disease_label = class_names[top_class_idx]
    confidence_score = float(np.max(probabilities) * 100)
    return disease_label, confidence_score


def get_treatment_plan(disease_name, confidence):
    client = genai.Client()

    prompt = f"""
            You are an expert agricultural pathology copilot. 
            A computer vision model diagnosed a crop leaf as '{disease_name}' with {confidence:.1f}% confidence.

            Provide a concise action plan formatted in clean markdown with the following 3 sections:
            1. Organic Remedies
            2. Chemical Treatments (only if infection is severe)
            3. Some more care tips

            Keep the advice practical, concise, and under 200 words total.
            """

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text

if __name__ == "__main__":
    image_path = sys.argv[1]

print(f"\nAnalyzing leaf image: {image_path} ...")
disease, confidence = predict_disease(image_path)

print(f"DIAGNOSIS: {disease}")
print(f"CONFIDENCE:      {confidence:.2f}%")

print("\nFetching treatment advisory from LLM...\n")
advisory = get_treatment_plan(disease, confidence)
print(advisory)