from stable_diffusion_inference import SDInference

config_path = "configs/stable-diffusion/v2-inference-v.yaml"
checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt"

model = SDInference(
    config_path=config_path,
    checkpoint_path=checkpoint_path
    )

model.predict_step("cats in hats")
