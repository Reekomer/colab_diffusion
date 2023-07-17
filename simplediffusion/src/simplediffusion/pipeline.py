from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from functools import lru_cache


@lru_cache(maxsize=None)
def build_diffusion_pipelines(
    model_path: Path,
    inpainting_model_path: Path,
    scheduler: bool,
    vae_model: AutoencoderKL,
) -> tuple:
    """
    Build a diffusion pipeline with a diffusion model as the base model.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="lpw_stable_diffusion",
        revision="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.safety_checker = None

    pipeimg = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipeimg.enable_xformers_memory_efficient_attention()
    pipeimg.safety_checker = None

    pipeinpainting = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_path,
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipeinpainting.enable_attention_slicing()
    pipeinpainting.enable_xformers_memory_efficient_attention()
    pipeinpainting.safety_checker = None
    if vae_model:
        pipe.vae = vae_model
        pipeimg.vae = vae_model
        pipeinpainting.vae = vae_model
    if scheduler:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipeimg.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeimg.scheduler.config
        )
        pipeinpainting.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeinpainting.scheduler.config
        )
    return pipe, pipeimg, pipeinpainting


@lru_cache(maxsize=None)
def build_control_net_pipelines(
    full_control_net_model_path: Path, full_diffusion_model_path: Path
) -> StableDiffusionControlNetPipeline:
    """
    Build a control net pipeline with a diffusion model as the base model.
    """
    controlnet = ControlNetModel.from_pretrained(
        full_control_net_model_path, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        full_diffusion_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe
