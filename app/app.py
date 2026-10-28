import os
import gradio as gr
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "mobilenetv2_finetune30_aug.keras")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Download mobilenetv2_finetune30_aug.keras (link in the README) into the models/ folder."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_damage(image):
    """Return {'Damaged': p, 'Undamaged': 1-p}. The model takes raw RGB 0..255 and has
    its own preprocessing and a single sigmoid output = P(damaged)."""
    if image is None:
        return {"Damaged": 0.0, "Undamaged": 0.0}
    image = image.convert("RGB").resize((224, 224))
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    p_damaged = float(model.predict(batch, verbose=0).ravel()[0])
    return {"Damaged": p_damaged, "Undamaged": 1.0 - p_damaged}

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