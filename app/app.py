import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "mobilenetv2_frozen_best.keras")
CLASS_NAMES = ["Undamaged", "Damaged"]

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Place mobilenetv2_frozen_best.keras inside the project's models/ folder."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_damage(image):
    if image is None:
        return {"Undamaged": 0.0, "Damaged": 0.0}
    image = image.convert("RGB").resize((224, 224))
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = np.asarray(model.predict(batch, verbose=0)[0]).ravel()
    if len(probabilities) != 2:
        raise ValueError(f"Expected 2 output probabilities, got shape {probabilities.shape}")
    return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(2)}

with gr.Blocks(title="Structural Damage Classifier") as demo:
    gr.Markdown("# Structural Damage Image Classifier")
    gr.Markdown(
        "Upload a structural image to estimate whether it appears damaged or undamaged. "
        "This is an academic demo, not a substitute for professional structural inspection."
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload a structural image")
        output = gr.Label(num_top_classes=2, label="Prediction and confidence")
    gr.Button("Classify Image").click(predict_damage, inputs=image_input, outputs=output)

if __name__ == "__main__":
   demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
