import streamlit as st
import numpy as np
from PIL import Image
from diffusion_composition.pipeline import DiffusionCompositionPipeline
from diffusion_composition.prompting import BoundingBoxPromptSetter
from diffusion_composition.utils import load_models
from diffusers import DiffusionPipeline

# Download the models and initialize the pipeline
models = load_models()
comp = DiffusionCompositionPipeline(vae=models['vae'], unet=models['unet'], scheduler=models['scheduler'])

# Set the page background color to blue
st.markdown(
    """
    <style>
        body {
            background-color: #3498db;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title and heading
st.title("Ad Frame Generator from Prompts")

# Input fields for prompts
background_prompt = st.text_input("Background Prompt", "the road of a bustling city street")
car_prompt = st.text_input("Car Prompt", "A toyota driving on the road")
person_prompt = st.text_input("Person Prompt", "A lady walking by")

# Button to generate image
if st.button("Generate Image"):
    # Create a BoundingBoxPromptSetter
    prompt_setter = BoundingBoxPromptSetter(text_encoder=models['text_encoder'], tokenizer=models['tokenizer'])

    # Set the prompts
    prompt_setter.set_background_prompt(background_prompt, guidance_scale=7.5)
    prompt_setter.add_local_prompt(car_prompt, top_margin=0.3, bottom_margin=0., left_margin=0.3, right_margin=0.4)
    prompt_setter.add_local_prompt(person_prompt, top_margin=0.4, bottom_margin=0., left_margin=0.8, right_margin=0.)
    prompt_setter.add_to_all_prompts("high quality photo")

    # Generate the image
    img = comp(prompt_setter=prompt_setter, num_inference_steps=50, bootstrap_steps=5, device='cuda', batch_size=6)
    img = Image.fromarray((img * 255).round().astype(np.uint8))

    # Display the generated image
    st.image(img, caption="Generated Image", use_column_width=True)

    
