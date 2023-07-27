from .sampling import *

class SDXL:
    def __init__(self, version: str="SDXL-base-1.0", mode="txt2img", add_pipeline=False) -> None:
        """
        version: VERSION2SPECS.keys()
        mode: ("txt2img", "img2img")
        add_pipeline: whether to Load SDXL-refiner
        """
        self.mode = mode
        self.version = version
        self.add_pipeline = add_pipeline
        self.version_dict = VERSION2SPECS[version]

        self.state = state = init_st(self.version_dict, load_filter=True)
        if state["msg"]:
            st.info(state["msg"])

    def __call__(self, prompt:str, ):
        state = self.state
        mode = self.mode
        add_pipeline = self.add_pipeline
        version = self.version
        version_dict=self.version_dict
        
        is_legacy = self.version_dict["is_legacy"]
        stage2strength = None
        finish_denoising = False

        if mode == "txt2img":
            out = run_txt2img(
                state,
                version,
                version_dict,
                is_legacy=is_legacy,
                return_latents=add_pipeline,
                filter=state.get("filter"),
                stage2strength=stage2strength,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
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