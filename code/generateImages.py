import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image

# Define the model and prompt
model_id = "runwayml/stable-diffusion-v1-5"
prompt = "A close-up of red strawberries in good lighting"

def generate_strawberry_images(prompt, num_images, size=(300, 300)):
    # Load the pre-trained Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Move model to GPU if available, else CPU
    
    if torch.cuda.is_available():
        device = "cuda" 
    elif torch.backends.mps.is_available():
        device = "mps" 
    else:
        device = "cpu"
    
    pipe.to(device)

    print(f"Generating {num_images} images on {device}...")

    # Generate multiple images
    for i in range(num_images):
        with autocast(device):  # Efficient inference with mixed precision
            image = pipe(prompt).images[0]

        # Resize the image to the specified size (300x300)
        image = image.resize(size, Image.LANCZOS)

        # Save the generated image with a unique filename
        save_path = f"sw_image1_{i+1}.png"
        image.save(save_path)
        print(f"Image {i+1} saved as {save_path}")

# Generate 50 strawberry images
generate_strawberry_images(prompt, num_images=85)
