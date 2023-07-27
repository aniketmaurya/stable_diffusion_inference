from .sampling import *


class SDXL:
    def __init__(self, checkpoint_path:str, version: str="SDXL-base-1.0", mode="txt2img", add_pipeline=False, low_vram=False, load_filter=True) -> None:
        """
        version: VERSION2SPECS.keys()
        mode: ("txt2img", "img2img")
        add_pipeline: whether to Load SDXL-refiner
        load_filter: DeepFloyd filter
        """
        self.checkpoint_path=checkpoint_path
        self.mode = mode
        self.version = version
        self.add_pipeline = add_pipeline
        self.version_dict = VERSION2SPECS[version]
        self.version_dict["ckpt"] = checkpoint_path

        self.state = state = init_st(self.version_dict, load_filter=load_filter)
        self.model = state["model"]
        self.filter = state.get("filter")
        set_lowvram_mode(low_vram)
        

    def __call__(self, prompt:str, negative_prompt=None):
        model = self.model
        filter = self.filter
        mode = self.mode
        add_pipeline = self.add_pipeline
        version = self.version
        version_dict=self.version_dict
        
        is_legacy = self.version_dict["is_legacy"]
        stage2strength = None
        finish_denoising = False

        out = run_txt2img(
            model,
            version,
            version_dict,
            prompt=prompt,
            negative_prompt=negative_prompt,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=stage2strength,
        )
        if isinstance(out, (tuple, list)):
            samples, samples_z = out
        else:
            samples = out
            samples_z = None

        images = []
        if samples is not None:
            for sample in samples:
                sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
                image = Image.fromarray(sample.astype(np.uint8))           
                images.append(image)
        return images
