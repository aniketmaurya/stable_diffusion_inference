from stable_diffusion_inference import create_text2image

text2image = create_text2image("sd1")

image = text2image("cats in hats", image_size=512, inference_steps=25)
image.save("cats in hats.png")
