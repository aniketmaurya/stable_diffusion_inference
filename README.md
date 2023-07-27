# Easy Stable Diffusion

Simple and easy stable diffusion inference with LightningModule on GPU, CPU and MPS (Possibly all devices supported by [Lightning](https://lightning.ai)).


## Installation

### Model variants

| Name     | Variant                          | Image Size |
|----------|----------------------------------|------------|
| sd1      | Stable Diffusion 1.5             | 512        |
| sd1.5    | Stable Diffusion 1.5             | 512        |
| sd1.4    | Stable Diffusion 1.4             | 512        |
| sd2_base | SD 2.0 trained on image size 512 | 512        |
| sd2_high | SD 2.0 trained on image size 768 | 768        |
| sdxl-base-1.0 | SDXL 1.0                    | 1024       |


### SDXL

```
pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"


pip install sgm @ git+https://github.com/Stability-AI/generative-models.git@main
pip install clip @ git+https://github.com/openai/CLIP.git
```


### To install SD 2.1 and earlier**

```
pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -q
pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
```

## Example

```python
from stable_diffusion_inference import create_text2image

# text2image = create_text2image("sd1")
# text2image = create_text2image("sd2_high")  # for SD 2.0 with 768 image size
text2image = create_text2image("sd2_base")  # for SD 2.0 with 512 image size

image = text2image("cats in hats", image_size=512, inference_steps=50)
image.save("cats in hats.png")
```

### Using SDXL 

```python
from stable_diffusion_inference import SDXL

checkpoint_path = "/data/aniket/stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
version = "SDXL-base-1.0"

text2image = SDXL(checkpoint_path=checkpoint_path,version=version, low_vram=True)
prompt = "Llama in a jungle, Lightning, AI themed, purple colors, detailed, 8k"
images = text2image(prompt)
images[0]
```
