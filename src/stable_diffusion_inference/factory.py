import os
import typing

from .lit_model import SDInference


def create_text2image(
        sd_variant: str,
        cache_dir: typing.Optional[str] = None,
        force_download: typing.Optional[bool] = None,
        **kwargs,
):
    model = None
    _ROOT_DIR = os.path.dirname(__file__)
    if sd_variant == "sd1.4":
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
    elif sd_variant == "sd1.5" or sd_variant == "sd1":
        config_path = f"{_ROOT_DIR}/configs/stable-diffusion/v1-inference.yaml"
        checkpoint_path = "https://storage.googleapis.com/gradsflow.appspot.com/oss/model-weights/v1-5-pruned-emaonly.ckpt"

        model = SDInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            version="1.4",
            cache_dir=cache_dir,
            force_download=force_download,
            ckpt_filename="v1-5-pruned-emaonly.ckpt",
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
