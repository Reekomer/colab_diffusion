import os
import torch
from PIL import Image
from typing import List
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    preprocess,
)
from colab_diffusion.prompt import PromptAttributes
import uuid
from tinydb import TinyDB
from diffusers import StableDiffusionImg2ImgPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def infer(
    prompt_attributes: PromptAttributes,
    num_samples: int,
    pipeimg: StableDiffusionImg2ImgPipeline,
) -> List[Image.Image]:
    with Image.open(prompt_attributes.init_image) as im:
        init_image = im.resize((prompt_attributes.width, prompt_attributes.height))
    init_image = init_image.convert("RGB")
    init_image = preprocess(init_image)
    init_images = [init_image] * num_samples
    generator = torch.Generator("cuda").manual_seed(prompt_attributes.seed)
    prompt = [prompt_attributes.prompt] * num_samples
    negative_prompt = [prompt_attributes.negative_prompt] * num_samples
    images = pipeimg(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_images,
        num_inference_steps=prompt_attributes.num_inference_steps,
        strength=prompt_attributes.strength,
        guidance_scale=prompt_attributes.guidance_scale,
        generator=generator,
    )["images"]
    return images


def save_images(images: List, base_path: str = "output/") -> List[str]:
    image_paths = []
    for image in images:
        image_path = os.path.join(base_path, f"{str(uuid.uuid4())}.png")
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths


def insert_in_db(
    prompt_attributes: PromptAttributes,
    database: TinyDB,
    image_paths: List,
    model_id: str,
    control_net_model_id: str = None,
) -> None:
    for path in image_paths:
        database.insert(
            {
                "id": str(uuid.uuid4()),
                "model_id": model_id,
                "control_net_id": control_net_model_id,
                "creation_date": prompt_attributes.creation_date.strftime(
                    "%d-%m-%Y_%H-%M-%S"
                ),
                "prompt": prompt_attributes.prompt,
                "negative_prompt": prompt_attributes.negative_prompt,
                "seed": prompt_attributes.seed,
                "process_type": prompt_attributes.process_type,
                "init_image": prompt_attributes.init_image,
                "width": prompt_attributes.width,
                "height": prompt_attributes.height,
                "strength": prompt_attributes.strength,
                "num_inference_steps": prompt_attributes.num_inference_steps,
                "guidance_scale": prompt_attributes.guidance_scale,
                "image_path": path,
            }
        )
