import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow logs

from flask import Flask, render_template, request, send_from_directory
from diffusers import StableDiffusionPipeline
import torch, datetime

# ─────────────────────────────
# Config
# ─────────────────────────────
SAVE_DIR = r"C:\Users\rrenu\OneDrive\Desktop\pixpopai\images"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────
# Load pipeline (only once)
# ─────────────────────────────
print("🔹 Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Memory optimizations
try:
    pipe.enable_attention_slicing()
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
except Exception as e:
    print("⚠️ Some memory optimizations not available:", e)

# ─────────────────────────────
# Flask app
# ─────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None
    if request.method == "POST":
        prompt = request.form.get("prompt")
        device_choice = request.form.get("device") or "cpu"
        mode_choice = request.form.get("pipe") or "normal"

        # Switch device if needed
        if device_choice == "cuda" and torch.cuda.is_available():
            pipe.to("cuda")
        else:
            pipe.to("cpu")

        # Adjust generation for fast mode
        gen_args = {}
        if mode_choice == "fast":
            gen_args = dict(num_inference_steps=15, height=512, width=512)

        # Generate the image
        image = pipe(prompt, **gen_args).images[0]

        # Save image with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{mode_choice}_image_{timestamp}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        image.save(filepath)

        image_url = f"/images/{filename}"

    return render_template("index.html", image_url=image_url)

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(SAVE_DIR, filename)

# ─────────────────────────────
# Entry point
# ─────────────────────────────
if __name__ == "__main__":
    # debug=True for dev; use_reloader=False prevents double pipeline load
    app.run(debug=True, use_reloader=False)
