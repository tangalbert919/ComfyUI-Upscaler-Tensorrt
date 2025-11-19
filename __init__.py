from .nodes.load_tensorrt_model import LoadUpscalerTensorrtModel
from .nodes.load_tensorrt_model_local import LoadUpscalerTensorrtModelLocal
from .nodes.upscaler_tensorrt import UpscalerTensorrt
from .nodes.tensorrt_settings import TensorrtSettings
from comfy_api.latest import ComfyExtension, io


class UpscalerTensorrtExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UpscalerTensorrt,
            LoadUpscalerTensorrtModel,
            LoadUpscalerTensorrtModelLocal,
            TensorrtSettings
        ]

async def comfy_entrypoint() -> UpscalerTensorrtExtension:
    return UpscalerTensorrtExtension()
