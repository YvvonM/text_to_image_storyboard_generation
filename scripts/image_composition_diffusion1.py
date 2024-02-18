


from diffusion_composition.pipeline import DiffusionCompositionPipeline
from diffusion_composition.prompting import BoundingBoxPromptSetter
from diffusion_composition.utils import load_models, upscale_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")

# Download the models and initialize the pipeline

models = load_models()


comp = DiffusionCompositionPipeline(vae=models['vae'],
                                   unet=models['unet'],
                                   scheduler=models['scheduler'])

# Set the prompts

prompt_setter = BoundingBoxPromptSetter(text_encoder=models['text_encoder'], tokenizer=models['tokenizer'])

prompt_setter.set_background_prompt("the road of a bustling city street", guidance_scale=7.5)

#Prompts set first are put in the foreground in case of intersections
prompt_setter.add_local_prompt("A toyota driving on the road",
                               top_margin=0.3,
                               bottom_margin=0.,
                               left_margin=0.3,
                               right_margin=0.4)


prompt_setter.add_local_prompt("A lady walking by",
                               top_margin=0.4,
                               bottom_margin=0.,
                               left_margin=0.8,
                               right_margin=0.)


prompt_setter.add_to_all_prompts("high quality photo")
fig = prompt_setter.draw_bboxes('bboxes.png')
plt.show(fig)

# Generate the image

img = comp(prompt_setter=prompt_setter,
           num_inference_steps=50,
           bootstrap_steps=5,
           device='cuda',
           batch_size=6)
img = Image.fromarray((img * 255).round().astype(np.uint8))
img.save("generated_image.png")
display(img)

#from matplotlib import pyplot as plt
#from text2scene.prompt_setting import BoundingBoxPromptSetter
# Create a BoundingBoxPromptSetter
prompt_setter = BoundingBoxPromptSetter(text_encoder=models['text_encoder'], tokenizer=models['tokenizer'])

# Set the background prompt
prompt_setter.set_background_prompt("A vibrant, bustling LEGO CITY set, showcasing key features like the police station, fire station, and various vehicles", guidance_scale=7.5)

# Add the 'Test Your Knowledge' prompt with adjusted margins for visibility
prompt_setter.add_local_prompt("A 'Test Your Knowledge' prompt in LEGO's signature yellow and red colors",
                               top_margin=0.05,
                               bottom_margin=0.65,
                               left_margin=0.1,
                               right_margin=0.1)

# Add the 'Tap to Start' button
prompt_setter.add_local_prompt("A 'Tap to Start' button",
                               top_margin=0.30,
                               bottom_margin=0.15,
                               left_margin=0.1,
                               right_margin=0.1)

# Add a common prompt to all, in this case, "high quality photo"
prompt_setter.add_to_all_prompts("high quality photo")

# Draw bounding boxes and show the figure
fig = prompt_setter.draw_bboxes('bboxes.png')
plt.show(fig)

img = comp(prompt_setter=prompt_setter,
           num_inference_steps=50,
           bootstrap_steps=5,
           device='cuda',
           batch_size=6)
img = Image.fromarray((img * 255).round().astype(np.uint8))
img.save("generated_image.png")
display(img)