from typing import List, Union
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from omegaconf import OmegaConf
import lightning as L
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> str:
        return self.prompts[i]


class LightningStableDiffusion(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
        size: int = 512,
    ):
        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.unet_config["params"]["use_fp16"] = False
        config.model.params.cond_stage_config["params"] = {"device": device}

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(state_dict, strict=False)

        self.sampler = DDIMSampler(self.model)

        self.initial_size = int(size / 8)
        self.steps = 50

        self.to(device)

    @torch.inference_mode()
    def predict_step(self, prompts: Union[str, List[str]], batch_idx: int):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, self.initial_size, self.initial_size]
            samples_ddim, _ = self.sampler.sample(
                S=self.steps,  # Number of inference steps, more steps -> higher quality
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results
