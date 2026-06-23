from comfy_api.latest import io
from ..utilities import LOAD_UPSCALER_NODE_CONFIG

IMAGE_DIM_MIN = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MIN")
IMAGE_DIM_OPT = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_OPT")
IMAGE_DIM_MAX = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MAX")

class TensorrtSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Advanced settings
        batch_size_defaults = io.Int.Input("batch_size_opt",
            default=1,
            min=1,
            max=100,
            step=1
        )
        height_defaults = io.Int.Input("height_opt",
            default=IMAGE_DIM_OPT,
            min=IMAGE_DIM_MIN,
            max=IMAGE_DIM_MAX,
            step=64
        )
        width_defaults = io.Int.Input("width_opt",
            default=IMAGE_DIM_OPT,
            min=IMAGE_DIM_MIN,
            max=IMAGE_DIM_MAX,
            step=64
        )
        batch_size_min = io.Int.Input("batch_size_min",
            default=1,
            min=1,
            max=100,
            step=1
        )
        height_min = io.Int.Input("height_min",
            default=IMAGE_DIM_MIN,
            min=IMAGE_DIM_MIN,
            max=IMAGE_DIM_MAX,
            step=64
        )
        width_min = io.Int.Input("width_min",
            default=IMAGE_DIM_MIN,
            min=IMAGE_DIM_MIN,
            max=IMAGE_DIM_MAX,
            step=64
        )
        batch_size_max = io.Int.Input("batch_size_max",
            default=1,
            min=1,
            max=100,
            step=1
        )
        height_max = io.Int.Input("height_max",
            default=IMAGE_DIM_MAX,
            min=IMAGE_DIM_MIN,
            max=IMAGE_DIM_MAX,
            step=64
        )
        width_max = io.Int.Input("width_max",
            default=IMAGE_DIM_MAX,
            min=IMAGE_DIM_MIN,
            max=IMAGE_DIM_MAX,
            step=64
        )
        weight_stream_option = io.Boolean.Input("weight_streaming",
                                                default=False)

        return io.Schema(
            node_id="TensorrtSettings",
            display_name="TensorRT Model Settings",
            category="TensorRT/upscaler",
            description="Set custom parameters for TensorRT model",
            inputs=[
                batch_size_min,
                batch_size_defaults,
                batch_size_max,
                height_min,
                height_defaults,
                height_max,
                width_min,
                width_defaults,
                width_max,
                weight_stream_option
            ],
            outputs=[
                io.Custom("trt_settings").Output("trt_settings")
            ]
        )
    
    @classmethod
    def execute(self, batch_size_min, batch_size_opt, batch_size_max,
                height_min, height_opt, height_max,
                width_min, width_opt, width_max,
                weight_streaming) -> io.NodeOutput:
        output = ([batch_size_min, batch_size_opt, batch_size_max],
                  [height_min, height_opt, height_max],
                  [width_min, width_opt, width_max],
                  weight_streaming)
        return io.NodeOutput(output)
