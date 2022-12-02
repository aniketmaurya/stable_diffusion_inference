import os
import tarfile
import typing
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any, List

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ldm1.models.diffusion.ddim import DDIMSampler as SD1Sampler
from ldm2.models.diffusion.ddim import DDIMSampler as SD2Sampler

from .data import PromptDataset

SAMPLERS = {"1.4": SD1Sampler, "2.0": SD2Sampler}


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

DOWNSAMPLING_FACTOR = 8
UNCONDITIONAL_GUIDANCE_SCALE = 9.0  # SD2 need higher than SD1 (~7.5)


def download_checkpoints(ckpt_path: str, cache_dir: typing.Optional[str] = None,
                         force_download: typing.Optional[bool] = None,
                         ckpt_filename: typing.Optional[str] = None) -> str:
    if ckpt_path.startswith("http"):
        # Ex: pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt
        ckpt_url = ckpt_path
        dest = str((Path(cache_dir) if cache_dir else Path()) / os.path.basename(ckpt_path))
        # Ex: ./512-base-ema.ckpt
        if dest.endswith(".tar.gz"):
            # Ex: https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/sd_weights.tar.gz
            ckpt_folder = dest.replace(".tar.gz", "")  # Ex: ./sd_weights
            Path(ckpt_folder).mkdir(parents=True, exist_ok=True)
            if not ckpt_filename:  # Ex: sd-v1-4.ckpt
                raise Exception("ckpt_filename must not be None")
            ckpt_path = str(Path(ckpt_folder) / ckpt_filename)  # Ex: ./sd_weights/sd-v1-4.ckpt
        else:
            ckpt_path = dest  # Ex: ./512-base-ema.ckpt
        if Path(ckpt_path).exists() and not force_download:
            return ckpt_path
        else:
            print("downloading checkpoints. This can take a while...")
            urllib.request.urlretrieve(ckpt_url, dest)
            if dest.endswith(".tar.gz"):
                file = tarfile.open(dest)
                file.extractall(ckpt_folder)
                file.close()
                os.unlink(dest)
            return ckpt_path
    else:
        return ckpt_path  # Ex: ./sd_weights/sd-v1-4.ckpt


def load_model_from_config(
    config: Any, ckpt: str, version: str, verbose: bool = False
) -> torch.nn.Module:
    if version == "2.0":
        from ldm2.util import instantiate_from_config

    elif version.startswith("1."):
        from ldm1.util import instantiate_from_config
    else:
        raise NotImplementedError(
            f"version={version} not supported. {SUPPORTED_VERSIONS}"
        )

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
        self, device: torch.device, config_path: str, checkpoint_path: str, version: str
    ):
        from omegaconf import OmegaConf

        if version == "2.0":
            SamplerCls = SAMPLERS[version]

        elif version.startswith("1."):
            SamplerCls = SAMPLERS[version]
        else:
            raise NotImplementedError(
                f"version={version} not supported. {SUPPORTED_VERSIONS}"
            )

        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.cond_stage_config["params"] = {"device": device}
        self.model = load_model_from_config(
            config, f"{checkpoint_path}", version=version
        )
        self.sampler = SamplerCls(self.model)

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
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        cache_dir: typing.Optional[str] = None,
        force_download: typing.Optional[bool] = None,
        ckpt_filename: typing.Optional[str] = None,
        accelerator: str = "auto",
        version="2.0",
    ):
        assert (
            version in SUPPORTED_VERSIONS
        ), f"supported version are {SUPPORTED_VERSIONS}"
        checkpoint_path = download_checkpoints(checkpoint_path, cache_dir, force_download, ckpt_filename)

        self.use_cuda: bool = torch.cuda.is_available() and accelerator in (
            "auto",
            "gpu",
        )
        precision = 16 if self.use_cuda else 32
        self.trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        device = self.trainer.strategy.root_device.type

        clear_cuda()
        self.model = StableDiffusionModule(
            device=device,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            version=version,
        )
        if self.use_cuda:
            self.model = self.model.to(torch.float16)
            clear_cuda()

    def __call__(
        self, prompts: List[str], image_size: int = 768, inference_steps: int = 50
    ) -> Image.Image:
        if isinstance(prompts, str):
            prompts = [prompts]
        trainer = self.trainer
        model = self.model

        img_dl = DataLoader(
            PromptDataset(prompts), batch_size=len(prompts), shuffle=False
        )
        model.predict_step = partial(
            model.predict_step,
            height=image_size,
            width=image_size,
            num_inference_steps=inference_steps,
        )
        pil_results = trainer.predict(model, dataloaders=img_dl)[0]
        if len(pil_results) == 1:
            return pil_results[0]
        return pil_results


def create_text2image(sd_variant: str, cache_dir: typing.Optional[str] = None,
                      force_download: typing.Optional[bool] = None, **kwargs):
    model = None
    _ROOT_DIR = os.path.dirname(__file__)
    if sd_variant == "sd1":
        config_path = f"{_ROOT_DIR}/configs/stable-diffusion/v1-inference.yaml"
        checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/sd_weights.tar.gz"

        model = SDInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            version="1.4",
            cache_dir=cache_dir,
            force_download=force_download,
            ckpt_filename="sd-v1-4.ckpt",
            **kwargs,
        )

    elif sd_variant == "sd2_high":
        config_path = f"{_ROOT_DIR}/configs/stable-diffusion/v2-inference-v.yaml"
        checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt"

        model = SDInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            force_download=force_download,
            version="2.0",
            **kwargs,
        )
    elif sd_variant == "sd2_base":
        config_path = f"{_ROOT_DIR}/configs/stable-diffusion/v2-inference.yaml"
        checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt"

        model = SDInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            force_download=force_download,
            version="2.0",
            **kwargs,
        )

    return model
