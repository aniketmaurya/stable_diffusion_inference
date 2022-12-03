# !pip install taming-transformers-rom1504 -q
# !pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
# !pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"

import gradio as gr
from functools import partial
import lightning as L
from lightning.app.components.serve import ServeGradio


class SDComparison(ServeGradio):
    inputs = gr.Textbox(label="Prompt", value="Cats in hats")
    outputs = [gr.Image(label="SD 1"), gr.Image(label="SD 2")]
    title = "Compare images from Stable Diffusion 1 and 2.0"

    def build_model(self):
        from stable_diffusion_inference import create_text2image

        sd1 = create_text2image("sd1")
        sd2 = create_text2image("sd2_base")  # for SD 2.0 with 512 image size
        return {"sd1": partial(sd1, image_size=512), "sd2": partial(sd2, image_size=512)}

    def predict(self, prompt: str):
        image1 = self.model["sd1"](prompts=prompt)
        image2 = self.model["sd2"](prompts=prompt)
        return [image1, image2]


component = L.LightningApp(SDComparison(cloud_compute=L.CloudCompute("gpu-fast", disk_size=40)))
