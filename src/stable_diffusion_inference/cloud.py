import lightning as L
from lightning.app.components.serve import PythonServer, Image
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class SDServe(PythonServer):
    def setup(self, *args, **kwargs) -> None:
        from stable_diffusion_inference import create_text2image
        self._model = create_text2image("sd1")

    def predict(self, request: Prompt):
        return {"image": self._model(request.prompt)}


component = SDServe(input_type=Prompt, output_type=Image)
app = L.LightningApp(component)
