import typing
import tarfile
import os
import urllib.request
from functools import partial
from typing import Any, List

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import PromptDataset
def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DOWNSAMPLING_FACTOR = 8
UNCONDITIONAL_GUIDANCE_SCALE = 9.0  # SD2 need higher than SD1 (~7.5)


def download_checkpoints(ckpt_path: str)-> str:
    "returns the path of model ckpt"
    dest = os.path.basename(ckpt_path)
    if ckpt_path.startswith("http"):
        if not os.path.exists(dest):
            print("downloading checkpoints. This can take a while...")
            urllib.request.urlretrieve(ckpt_path, dest)
        else: print(f"model already exists {dest}")

        if ckpt_path.endswith("tar.gz"):
            file = tarfile.open(dest)
            target_file = dest.replace(".tar.gz", "")
            file.extractall(target_file)
            return target_file
        return dest
    return ckpt_path


def load_model_from_config(
    config: Any, ckpt: str, version:str, verbose: bool = False
) -> torch.nn.Module:
    if version == "2.0":
        from sd2.ldm.util import instantiate_from_config
    
    elif version.startswith("1."):
        from sd1.ldm.util import instantiate_from_config
    else:
        raise NotImplementedError(f"version={version} not supported. {SUPPORTED_VERSIONS}")

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


class StableDiffusionModule(L.LightningModule):
    def __init__(
        self,
        device: torch.device,
        config_path: str,
        checkpoint_path: str,
        version: str
    ):
        from omegaconf import OmegaConf
        if version == "2.0":
            from sd2.ldm.models.diffusion.ddim import DDIMSampler
        
        elif version.startswith("1."):
            from sd1.ldm.models.diffusion.ddim import DDIMSampler
        else:
            raise NotImplementedError(f"version={version} not supported. {SUPPORTED_VERSIONS}")

        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.cond_stage_config["params"] = {"device": device}
        self.model = load_model_from_config(config, f"{checkpoint_path}", version=version)
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


SUPPORTED_VERSIONS = {"1.4", "1.5", "2.0"}


class SDInference:
    """
    version: supported version are 1.5 and 2.0
    """
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        accelerator: str="auto",
        version="2.0",
    ):
        assert version in SUPPORTED_VERSIONS, f"supported version are {SUPPORTED_VERSIONS}"
        checkpoint_path = download_checkpoints(checkpoint_path)

        precision = 16 if torch.cuda.is_available() else 32
        self.trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        device=self.trainer.strategy.root_device.type
        
        clear_cuda()
        self.model = StableDiffusionModule(
            device=device, checkpoint_path=checkpoint_path, config_path=config_path, version=version
        )
        if torch.cuda.is_available():
            self.model = self.model.to(torch.float16)
        clear_cuda()
        

    def __call__(self, prompts: List[str], image_size: int=768, inference_steps:int = 50)-> Image.Image:
        if isinstance(prompts, str):
            prompts = [prompts]
        trainer = self.trainer
        model = self.model

        img_dl = DataLoader(
            PromptDataset(prompts), batch_size=len(prompts), shuffle=False
        )
        model.predict_step = partial(model.predict_step, height=image_size, width=image_size, num_inference_steps=inference_steps)
        pil_results = trainer.predict(model, dataloaders=img_dl)[0]
        if len(pil_results)==1:
            return pil_results[0]
        return pil_results

def create_text2image(sd_variant: str):
    model = None
    if sd_variant=="sd1":
        config_path = "configs/stable-diffusion/v1-inference.yaml"
        checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/sd_weights.tar.gz"

        dest = download_checkpoints(checkpoint_path)
        checkpoint_path = dest + "/sd-v1-4.ckpt"

        model = SDInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            version="1.4"
            )

    elif sd_variant=="sd2":
        config_path = "configs/stable-diffusion/v2-inference-v.yaml"
        checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt"

        model = SDInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            version="2.0"
            )

    return model
