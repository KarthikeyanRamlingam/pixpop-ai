from diffusers import StableDiffusionPipeline
import torch
import os
import webbrowser
import datetime

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# If you have a GPU, this makes it faster:
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# 🔹 Ask user for prompt
prompt = input("🖊️ Enter your image prompt: ")

# Generate
print("🎨 Generating image, please wait...")
image = pipe(prompt).images[0]

# Define save folder
save_dir = r"C:\Users\rrenu\OneDrive\Desktop\pixpopai\images"

# Make sure the folder exists
os.makedirs(save_dir, exist_ok=True)

# 🔹 Create a unique filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"image_{timestamp}.png"
filepath = os.path.join(save_dir, filename)

# Save image
image.save(filepath)
print(f"✅ Image generated: {filename}")

# Show full path
filepath = os.path.abspath(filename)
print("📂 Image saved at:{filepath}")

# Open automatically
webbrowser.open(filepath)
