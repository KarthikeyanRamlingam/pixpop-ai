from diffusers import StableDiffusionPipeline
import torch
import os
import webbrowser
import datetime

#path
model_path=r"D:\pixpop\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
# 🔹 Load a lighter / faster model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Force CPU (since Intel UHD is not supported for CUDA)
pipe = pipe.to("cpu")

# Ask user for prompt
prompt = input("🖊️ Enter your image prompt: ")

print("🎨 Generating image, please wait... (optimized for CPU)")

# 🔹 Use fewer steps + smaller image size
image = pipe(
    prompt,
    num_inference_steps=15,  # Faster (instead of 50)
    height=512,              # Smaller image (instead of 512)
    width=512
).images[0]

# Define save folder
save_dir = r"C:\Users\rrenu\OneDrive\Desktop\pixpopai\images"

# Make sure the folder exists
os.makedirs(save_dir, exist_ok=True)

# Save with unique timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"fast_image_{timestamp}.png"
filepath = os.path.join(save_dir, filename)
image.save(filepath)

# Show path
filepath = os.path.abspath(filename)
print(f"✅ Image generated: {filename}")
print(f"📂 Saved at: {filepath}")

# Open automatically
webbrowser.open(filepath)
