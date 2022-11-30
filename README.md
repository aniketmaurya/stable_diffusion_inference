# Easy Stable Diffusion

Simple and easy stable diffusion inference on CPU, GPU, MPS and all the devices supported by Lightning.

```python
from stable_diffusion_inference import create_text2image

text2image = create_text2image("sd1")
# text2image = create_text2image("sd2")  # for SD 2.0

image = text2image("cats in hats", image_size=512, inference_steps=50)
image.save("cats in hats.png")

```
