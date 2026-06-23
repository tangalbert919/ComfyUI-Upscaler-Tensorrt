import os
import folder_paths
import torch
from comfy_api.latest import io
from ..trt_utilities import Engine
from ..utilities import download_file, logger, LOAD_UPSCALER_NODE_CONFIG
import comfy.model_management as mm
import time

# Support TensorRT-RTX
TENSORRT_RTX_AVAILABLE = False
import importlib
if importlib.util.find_spec('tensorrt_rtx') is not None:
    import tensorrt_rtx as trt
    TENSORRT_RTX_AVAILABLE = True
    logger.info("Using TensorRT RTX")
else:
    import tensorrt as trt

IMAGE_DIM_MIN = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MIN")
IMAGE_DIM_OPT = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_OPT")
IMAGE_DIM_MAX = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MAX")

class LoadUpscalerTensorrtModelLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")

        return io.Schema(
            node_id="LoadUpscalerTensorrtModelLocal",
            display_name="Convert Upscaler Model to TensorRT",
            category="TensorRT/upscaler",
            description="Convert Upscaler Model to TensorRT",
            inputs=[
                io.UpscaleModel.Input("upscaler_model"),
                io.String.Input("filename", default="model"),
                io.Combo.Input("precision", options=precision_options,
                               default=precision_default),
                io.Custom("trt_settings").Input("trt_settings", optional=True)
            ],
            outputs=[
                io.Custom("upscaler_trt_model").Output("upscaler_trt_model")
            ]
        )

    # Check dynamic shapes for esrgan 4x model
    @classmethod
    def supports_dynamic_shapes_esrgan(self, model, scale=2):
        input_shapes = [
        (1, 3, 64, 64),
        (1, 3, 128, 128),
        (1, 3, 256, 192),
        (1, 3, 512, 256),
        (1, 3, 512, 512)
        ]

        all_passed = True

        with torch.no_grad():
            for shape in input_shapes:
                try:
                    dummy_input = torch.randn(*shape).cuda()
                    output = model(dummy_input)

                    expected_h = shape[2] * scale
                    expected_w = shape[3] * scale

                    assert output.shape[0] == shape[0], "Batch size mismatch"
                    assert output.shape[1] == shape[1], "Channel mismatch"
                    assert output.shape[2] == expected_h, f"Height mismatch: expected {expected_h}, got {output.shape[2]}"
                    assert output.shape[3] == expected_w, f"Width mismatch: expected {expected_w}, got {output.shape[3]}"

                    print(f"Success: input {shape} → output {output.shape}")
                except Exception as e:
                    all_passed = False
                    print(f"Failure: input {shape} → error: {e}")
                    torch.cuda.empty_cache()

        if all_passed: print(f"Success: Dynamic shapes supported.")
        if not all_passed: print(f"Failure: Dynamic shapes NOT supported.")
        return all_passed

    @classmethod
    def execute(self, upscaler_model, filename, precision, trt_settings) -> io.NodeOutput:
            batch = trt_settings[0] if trt_settings is not None else [1, 1, 1]
            height = trt_settings[1] if trt_settings is not None else [IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX]
            width = trt_settings[2] if trt_settings is not None else [IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX]
            tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
            onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

            os.makedirs(tensorrt_models_dir, exist_ok=True)
            os.makedirs(onnx_models_dir, exist_ok=True)

            onnx_model_path = os.path.join(onnx_models_dir, f"{filename}_{precision}.onnx")
            if not os.path.exists(onnx_model_path):
                logger.info(f"Converting model to {filename}_{precision}.onnx")
                device = mm.get_torch_device()
                mm.free_memory(mm.module_size(upscaler_model.model), device)
                upscaler_model.to(device)

                if self.supports_dynamic_shapes_esrgan(upscaler_model.model, scale=upscaler_model.scale):
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

            engine_channel = 3
            engine_min_batch, engine_opt_batch, engine_max_batch = batch
            engine_min_h, engine_opt_h, engine_max_h = height
            engine_min_w, engine_opt_w, engine_max_w = width
            tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{onnx_model_path.split('/')[-1].split('.')[0]}_{precision if not TENSORRT_RTX_AVAILABLE else 'rtx'}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{trt.__version__}.trt")

            if not os.path.exists(tensorrt_model_path):
                if not os.path.exists(onnx_model_path):
                    onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{onnx_model_path.split('/')[-1].split('.')[0]}.onnx"
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
                    weight_streaming=trt_settings[3] if trt_settings is not None else False
                )
                e = time.time()
                logger.info(f"Time taken to build: {(e-s)} seconds")

            logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
            mm.soft_empty_cache()
            engine = Engine(tensorrt_model_path)
            engine.load()
            engine.model_name = onnx_model_path.split('/')[-1].split('.')[0]

            return io.NodeOutput(engine)
