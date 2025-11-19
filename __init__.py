import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from comfy_api.latest import ComfyExtension, io
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
from .scripts.export_onnx import supports_dynamic_shapes_esrgan
import comfy.model_management as mm
import time
import json # <--- Import json module

# Support TensorRT-RTX
TENSORRT_RTX_AVAILABLE = False
import importlib
if importlib.util.find_spec('tensorrt_rtx') is not None:
    import tensorrt_rtx as tensorrt
    TENSORRT_RTX_AVAILABLE = True
else:
    import tensorrt

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 768
IMAGE_DIM_MAX = 4096

# --- Function to determine scaling factor from model name ---
def get_scale_factor_from_model_name(model_name: str) -> int:
    lower_name = model_name.lower()
    if "1x" in lower_name or "x1" in lower_name:
        return 1
    elif "2x" in lower_name or "x2" in lower_name:
        return 2
    else:
        return 4  # Default

# --- Function to load configuration ---
def load_node_config(config_filename="load_upscaler_config.json"):
    """Loads node configuration from a JSON file."""
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)
    
    default_config = { # Fallback in case file is missing or corrupt
        "model": {
            "options": ["4x-UltraSharp"],
            "default": "4x-UltraSharp",
            "tooltip": "Default model (fallback from code)"
        },
        "precision": {
            "options": ["fp16", "fp32"],
            "default": "fp16",
            "tooltip": "Default precision (fallback from code)"
        }
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_filename}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config

# --- Load the configuration once when the module is imported ---
LOAD_UPSCALER_NODE_CONFIG = load_node_config()


class UpscalerTensorrt(io.ComfyNode):
    @classmethod
    def define_schema(s) -> io.Schema:
        return io.Schema(
            node_id="UpscalerTensorrt",
            display_name="Upscaler TensorRT ⚡",
            category="TensorRT/upscaler",
            description="Upscale images with TensorRT",
            inputs=[
                io.Image.Input("images"),
                io.Custom("upscaler_trt_model").Input("upscaler_trt_model"),
                io.Combo.Input("resize_to",
                               options=["none", "HD", "FHD", "2k", "4k", "2x", "3x"],
                               tooltip="Resize the upscaled image to fixed resolutions, optional")
            ],
            outputs=[
                io.Image.Output()
            ]
        )

    @classmethod
    def execute(self, images, upscaler_trt_model, resize_to) -> io.NodeOutput:
        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        for dim in (H, W):
            if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
                raise ValueError(f"Input image dimensions fall outside of the supported range: {IMAGE_DIM_MIN} to {IMAGE_DIM_MAX} px!\nImage dimensions: {W}px by {H}px")

        final_width, final_height = get_final_resolutions(W, H, resize_to)

        # --- Determination of the scaling factor ---
        try:
            scale_factor = get_scale_factor_from_model_name(upscaler_trt_model.model_name)
        except AttributeError:
            logger.warning("Attribute ‘model_name’ not found. Default 4x used.")
            scale_factor = 4

        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*scale_factor}, W:{W*scale_factor} | Final resolution: H:{final_height}, W:{final_width} | resize_to: {resize_to}")

        device = mm.get_torch_device()

        memory_required = upscaler_trt_model.get_memory_size()
        memory_required += (H * W * 3) * images.element_size() * scale_factor
        memory_required += images.nelement() * images.element_size()
        mm.free_memory(memory_required, device)

        # Do split batching if input batch size exceeds engine's max
        min_batch = upscaler_trt_model.get_min_batch_size()
        max_batch = upscaler_trt_model.get_max_batch_size()
        for i in range(max_batch, min_batch - 1, -1):
            if B % i == 0:
                curr_split_batch = B // i
                break

        shape_dict = {
            "input": {"shape": (min(max_batch, B), 3, H, W)},
            "output": {"shape": (min(max_batch, B), 3, H*scale_factor, W*scale_factor)},
        }

        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=min(max_batch, B)))

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device())
        must_resize = W*scale_factor != final_width or H*scale_factor != final_height

        for batch in range(curr_split_batch):
            result = upscaler_trt_model.infer({"input": images_list[batch]}, cudaStream)
            result = result["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result, 
                    size=(final_height, final_width),
                    mode='bicubic',
                    antialias=True
                )

            for output_index, upscaled_index in enumerate(range(batch*min(max_batch, B), batch*min(max_batch, B) + len(images_list[batch]))):
                upscaled_frames[upscaled_index] = result[output_index].to(mm.intermediate_device())
                pbar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)
        upscaler_trt_model.reset()
        mm.soft_empty_cache()

        logger.info(f"Output shape: {output.shape}")
        return io.NodeOutput(output)

class LoadUpscalerTensorrtModelBase(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema: # Changed 's' to 'cls' for convention
        raise NotImplementedError

    @classmethod
    def _load_upscaler_tensorrt_model(self, model, precision, batch, height, width, weight_streaming):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")
        
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = batch
        engine_min_h, engine_opt_h, engine_max_h = height
        engine_min_w, engine_opt_w, engine_max_w = width
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision if not TENSORRT_RTX_AVAILABLE else 'rtx'}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                logger.info(f"Onnx model found at: {onnx_model_path}")

            logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" and not TENSORRT_RTX_AVAILABLE else False,
                input_profile=[
                    {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_opt_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]},
                ],
                weight_streaming=weight_streaming
            )
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        # --- Save model name for later access ---
        engine.model_name = model

        return engine

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

class LoadUpscalerTensorrtModelAdvanced(LoadUpscalerTensorrtModelBase):

    @classmethod
    def define_schema(cls) -> io.Schema: # Changed 's' to 'cls' for convention
        # Use the pre-loaded configuration
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})

        # Provide sensible defaults if keys are missing in the config (though load_node_config handles this broadly)
        model_options = model_config.get("options", ["4x-UltraSharp"])
        model_default = model_config.get("default", "4x-UltraSharp")
        model_tooltip = model_config.get("tooltip", "Select a model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

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
            node_id="LoadUpscalerTensorrtAdvanced",
            display_name="Load Upscale TensorRT Model (Advanced)",
            category="TensorRT/upscaler",
            description="Load TensorRT model (advanced)",
            inputs=[
                io.Combo.Input("model", options=model_options,
                               default=model_default,
                               tooltip=model_tooltip),
                io.Combo.Input("precision", options=precision_options,
                               default=precision_default,
                               tooltip=precision_tooltip),
                batch_size_min,
                batch_size_defaults,
                batch_size_max,
                height_min,
                height_defaults,
                height_max,
                width_min,
                width_defaults,
                width_max,
                weight_stream_option,
            ],
            outputs=[
                io.Custom("upscaler_trt_model").Output("upscaler_trt_model")
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(self, model, precision, batch_size_min,
                                     batch_size_opt, batch_size_max, height_min,
                                     height_opt, height_max, width_min,
                                     width_opt, width_max, weight_streaming) -> io.NodeOutput:
        batch = [batch_size_min, batch_size_opt, batch_size_max]
        height = [height_min, height_opt, height_max]
        width = [width_min, width_opt, width_max]
        return io.NodeOutput(super()._load_upscaler_tensorrt_model(model, precision, batch, height, width, weight_streaming))

class LoadUpscalerTensorrtModel(LoadUpscalerTensorrtModelBase):

    @classmethod
    def define_schema(cls) -> io.Schema: # Changed 's' to 'cls' for convention
        # Use the pre-loaded configuration
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})

        # Provide sensible defaults if keys are missing in the config (though load_node_config handles this broadly)
        model_options = model_config.get("options", ["4x-UltraSharp"])
        model_default = model_config.get("default", "4x-UltraSharp")
        model_tooltip = model_config.get("tooltip", "Select a model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        return io.Schema(
            node_id="LoadUpscalerTensorrtModel",
            display_name="Load Upscale TensorRT Model",
            category="TensorRT/upscaler",
            description="Load TensorRT model",
            inputs=[
                io.Combo.Input("model", options=model_options,
                               default=model_default,
                               tooltip=model_tooltip),
                io.Combo.Input("precision", options=precision_options,
                               default=precision_default,
                               tooltip=precision_tooltip),
                io.Custom("trt_settings").Input("trt_settings", optional=True)
            ],
            outputs=[
                io.Custom("upscaler_trt_model").Output("upscaler_trt_model")
            ]
        )

    @classmethod
    def execute(self, model, precision, trt_settings) -> io.NodeOutput:
        batch = trt_settings[0] if trt_settings is not None else [1, 1, 1]
        height = trt_settings[1] if trt_settings is not None else [IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX]
        width = trt_settings[2] if trt_settings is not None else [IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX]
        weight_streaming = trt_settings[3] if trt_settings is not None else False
        return io.NodeOutput(super()._load_upscaler_tensorrt_model(model, precision, batch, height, width, weight_streaming))

class LoadUpscalerTensorrtModelLocal(LoadUpscalerTensorrtModelBase):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Use the pre-loaded configuration
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})

        # Provide sensible defaults if keys are missing in the config (though load_node_config handles this broadly)
        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        return io.Schema(
            node_id="LoadUpscalerTensorrtModelLocal",
            display_name="Convert Upscaler Model to TensorRT",
            category="TensorRT/upscaler",
            description="Convert Upscaler Model to TensorRT",
            inputs=[
                io.UpscaleModel.Input("upscaler_model"),
                io.String.Input("filename", default="model"),
                io.Combo.Input("precision", options=precision_options,
                               default=precision_default,
                               tooltip=precision_tooltip),
                io.Custom("trt_settings").Input("trt_settings", optional=True)
            ],
            outputs=[
                io.Custom("upscaler_trt_model").Output("upscaler_trt_model")
            ]
        )

    @classmethod
    def execute(self, upscaler_model, filename, precision, trt_settings) -> io.NodeOutput:
        batch = trt_settings[0] if trt_settings is not None else [1, 1, 1]
        height = trt_settings[1] if trt_settings is not None else [IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX]
        width = trt_settings[2] if trt_settings is not None else [IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX]
        weight_streaming = trt_settings[3] if trt_settings is not None else False

        # Convert to ONNX first
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{filename}_{precision}.onnx")
        if not os.path.exists(onnx_model_path):
            logger.info(f"Converting model to {filename}_{precision}.onnx")
            device = mm.get_torch_device()
            mm.free_memory(mm.module_size(upscaler_model.model), device)
            upscaler_model.to(device)

            if supports_dynamic_shapes_esrgan(upscaler_model.model, scale=upscaler_model.scale):
                shape = (1, 3, 64, 64)
            else:
                shape = (1, 3, 512, 512)
            x = torch.rand(*shape).to(device)

            dynamic_axes = {
                "input": {0: "batch_size", 2: "width", 3: "height"},
                "output": {0: "batch_size", 2: "width", 3: "height"},
            }

            if precision == "fp16":
                x = x.to(dtype=torch.float16)
                upscaler_model.model.to(dtype=torch.float16)

            with torch.no_grad():
                torch.onnx.export(
                    upscaler_model.model,
                    x,
                    onnx_model_path,
                    input_names=['input'],
                    output_names=['output'],
                    export_params=True,
                    dynamic_axes=dynamic_axes,
                    external_data=False,
                )
        else:
            logger.info("ONNX found, no need to convert")
        return io.NodeOutput(super()._load_upscaler_tensorrt_model(onnx_model_path.split('/')[-1].split('.')[0],
                                                                   precision, batch, height, width, weight_streaming))

class UpscalerTensorrtExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UpscalerTensorrt,
            LoadUpscalerTensorrtModel,
            LoadUpscalerTensorrtModelAdvanced,
            LoadUpscalerTensorrtModelLocal,
            TensorrtSettings
        ]

async def comfy_entrypoint() -> UpscalerTensorrtExtension:
    return UpscalerTensorrtExtension()
