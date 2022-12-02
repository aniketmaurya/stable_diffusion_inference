# !pip install taming-transformers-rom1504 -q
# !pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
# !pip install gradio==3.6
# !pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"

from functools import partial

import gradio as gr
import lightning as L
from lightning.app.components.serve import ServeGradio


class SDComparison(ServeGradio):
    inputs = gr.Textbox(label="Prompt", value="Cats in hats")
    outputs = [gr.Image(label="SD 1"), gr.Image(label="SD 2")]

    def build_model(self):
        return {}
        # from stable_diffusion_inference import create_text2image

        # sd1 = create_text2image("sd1")
        # sd2 = create_text2image("sd2_base")  # for SD 2.0 with 512 image size
        # return {"sd1": partial(sd1, image_size=512), "sd2": partial(sd2, image_size=512)}

    def predict(self, prompt: str):
        image1 = self.model["sd1"](prompt=prompt)
        image2 = self.model["sd2"](prompt=prompt)
        return [image1, image2]


component = L.LightningApp(SDComparison(cloud_compute=L.CloudCompute("gpu-fast", disk_size=40)))
