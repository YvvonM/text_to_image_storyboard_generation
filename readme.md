# Ad Composition with Diffusion Models

This repository contains Python code for generating advertisement compositions using Diffusion Models. The code leverages the DiffusionCompositionPipeline from the Diffusion Composition library.

## Installation

1. Install the required dependencies:

    ```bash
    pip install git+https://github.com/GabrieleSgroi/image_composition_diffusion
    ```
  You can also run the following line of code 

    ```
    pip install -r requirements.txt
    ```


2. Download the models:

    ```python
    from diffusion_composition.utils import load_models

    models = load_models()
    ```

3. Run the code:

    ```bash
    python your_script_name.py
    ```

## Usage

The provided script generates advertisement compositions based on prompts. It uses a DiffusionCompositionPipeline to combine various elements and generate visually appealing images.

1. Set up your prompts:

    ```python
    from diffusion_composition.prompting import BoundingBoxPromptSetter

    prompt_setter = BoundingBoxPromptSetter(text_encoder=models['text_encoder'], tokenizer=models['tokenizer'])
    prompt_setter.set_background_prompt("the road of a bustling city street", guidance_scale=7.5)
    # Add more prompts as needed
    ```

2. Generate the image:

    ```python
    img = comp(prompt_setter=prompt_setter, num_inference_steps=50, bootstrap_steps=5, device='cuda', batch_size=6)
    ```

3. Save and display the generated image:

    ```python
    img.save("generated_image.png")
    display(img)
    ```

## Examples

The provided examples showcase different prompts for generating advertisement compositions, demonstrating the flexibility of the Diffusion Composition approach.

Feel free to modify the prompts and explore various creative possibilities!

## User UI
This model also contains a user interface developed using streamlit. To run the interface, run the following line of code in your terminal

```
streamlit run streamlit_app.py
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
