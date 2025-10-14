import os
import numpy as np
import torch
from comfy.utils import ProgressBar
from comfy_api.latest import io
from ..trt_utilities import Engine
from ..utilities import logger, get_final_resolutions, get_model_scale, LOAD_UPSCALER_NODE_CONFIG
import comfy.model_management as mm
from tqdm import tqdm

IMAGE_DIM_MIN = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MIN")
IMAGE_DIM_OPT = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_OPT")
IMAGE_DIM_MAX = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MAX")

class UpscalerTensorrt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UpscalerTensorrt",
            display_name="Upscaler TensorRT ⚡",
            category="TensorRT/upscaler",
            description="Upscale images with TensorRT",
            inputs=[
                io.Image.Input("images"),
                io.Custom("upscaler_trt_model").Input("upscaler_trt_model"),
                io.Combo.Input("resize_to",
                               options=["none", "custom", "HD", "FHD", "2k", "4k",
                                        "1x", "1.5x", "2x", "2.5x", "3x", "3.5x",
                                        "4x", "5x", "6x", "7x", "8x", "9x", "10x"],
                               tooltip="Resize the upscaled image to fixed resolutions, optional"),
                io.Int.Input("resize_width", default=1024, min=1, max=8192),
                io.Int.Input("resize_height", default=1024, min=1, max=8192)
            ],
            outputs=[
                io.Image.Output()
            ]
        )

    @classmethod
    def execute(self, **kwargs) -> io.NodeOutput:
        images = kwargs.get("images")
        upscaler_trt_model = kwargs.get("upscaler_trt_model")
        resize_to = kwargs.get("resize_to")

        model_name = getattr(upscaler_trt_model, "model_name", None)

        scale = get_model_scale(model_name)

        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        for dim in (H, W):
            if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
                raise ValueError("Image size out of bounds")

        if resize_to == "custom":
            final_width = kwargs.get("resize_width")
            final_height = kwargs.get("resize_height")
        elif resize_to == "none":
            final_width = W * scale
            final_height = H * scale
        else:
            final_width, final_height = get_final_resolutions(W, H, resize_to, scale)

        logger.info(f"Scale: {scale} | {H}x{W} -> {H*scale}x{W*scale} | final {final_height}x{final_width}")

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H * scale, W * scale)},
        }

        device = mm.get_torch_device()

        memory_required = upscaler_trt_model.get_memory_size()
        memory_required += (H * W * 3) * images.element_size() * scale
        memory_required += images.nelement() * images.element_size()
        mm.free_memory(memory_required, device)

        # Do split batching if input batch size exceeds engine's max
        min_batch = upscaler_trt_model.get_min_batch_size()
        max_batch = upscaler_trt_model.get_max_batch_size()
        for i in range(max_batch, min_batch - 1, -1):
            if B % i == 0:
                curr_split_batch = B // i
                break

        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)

        images_list = list(torch.split(images_bchw, split_size_or_sections=min(max_batch, B)))

        upscaled_frames = torch.empty(
            (B, C, final_height, final_width),
            dtype=torch.float32,
            device=mm.intermediate_device()
        )

        must_resize = (W * scale != final_width or H * scale != final_height)

        bar_format = "[\033[94mComfyUI-Upscaler-Tensorrt\033[0m|\033[92mINFO\033[0m] - \033[92m{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        progress_bar = tqdm(
            total=B,
            desc="Upscaling",
            bar_format=bar_format,
            disable=(B == 1)
        )

        for batch in range(curr_split_batch):
            result = upscaler_trt_model.infer({"input": images_list[batch]}, cudaStream)["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result,
                    size=(final_height, final_width),
                    mode="bicubic",
                    antialias=True
                )

            for output_index, upscaled_index in enumerate(range(batch*min(max_batch, B), batch*min(max_batch, B) + len(images_list[batch]))):
                upscaled_frames[upscaled_index] = result[output_index].to(mm.intermediate_device())
                pbar.update(1)
            progress_bar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)

        upscaler_trt_model.reset()
        mm.soft_empty_cache()

        return io.NodeOutput(output)