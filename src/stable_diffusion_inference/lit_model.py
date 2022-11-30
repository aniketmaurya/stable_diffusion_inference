import typing
import os
import urllib.request
from functools import partial
from typing import Any, List

import lightning as L
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from .data import PromptDataset

DOWNSAMPLING_FACTOR = 8
UNCONDITIONAL_GUIDANCE_SCALE = 9.0  # SD2 need higher than SD1 (~7.5)


def load_model_from_config(
    config: Any, ckpt: str, verbose: bool = False
) -> torch.nn.Module:
    from ldm.util import instantiate_from_config

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


class StableDiffusionModule(LightningModule):
    def __init__(
        self,
        device: torch.device,
        config_path: str,
        checkpoint_path: str,
        version: str
    ):  
        if version == "2.0":
            from sd2.ldm.models.diffusion.ddim import DDIMSampler
            from omegaconf import OmegaConf

        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.cond_stage_config["params"] = {"device": device}
        self.model = load_model_from_config(config, f"{checkpoint_path}")
        self.sampler = DDIMSampler(self.model)

    @typing.no_type_check
    @torch.inference_mode()
    def predict_step(
        self,
        prompts: List[str],
        batch_idx: int,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> Any:
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR]
            samples_ddim, _ = self.sampler.sample(
                S=num_inference_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]

        return pil_results


SUPPORTED_VERSIONS = {"1.5", "2.0"}

def download_checkpoints(ckpt_path: str)-> str:
    "returns the path of model ckpt"
    dest = os.path.basename(ckpt_path)
    if ckpt_path.startswith("http"):
        urllib.request.urlretrieve(ckpt_path, dest)
        return dest
    return ckpt_path


class Text2Image:
    """
    version: supported version are 1.5 and 2.0
    """
    def __init__(
        self,
        device: torch.device,
        config_path: str,
        checkpoint_path: str,
        version="1.5",
    ):
        assert version in SUPPORTED_VERSIONS, f"supported version are {SUPPORTED_VERSIONS}"
        checkpoint_path = download_checkpoints(checkpoint_path)
        
        self.model = StableDiffusionModule(
            device=device, checkpoint_path=checkpoint_path, config_path=config_path, version=version
        )
        precision = 16 if torch.cuda.is_available() else 32
        self.trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

    def __call__(self, prompts: List[str], image_size: int=768, inference_steps:int = 50)-> Image.Image:
        trainer = self.trainer
        model = self.model

        img_dl = DataLoader(
            PromptDataset(prompts), batch_size=len(prompts), shuffle=False
        )
        model.predict_step = partial(model.predict_step, height=image_size, width=image_size, num_inference_steps=inference_steps)
        pil_results = trainer.predict(model, dataloaders=img_dl)[0]
        return pil_results
