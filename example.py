from stable_diffusion_inference import SDInference

config_path = "configs/stable-diffusion/v2-inference-v.yaml"
checkpoint_path = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt"

text2image = SDInference(
    config_path=config_path,
    checkpoint_path=checkpoint_path
    )

image = text2image("cats in hats", image_size=512)
image.save("cats in hats.png")
