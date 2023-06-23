from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, LineartAnimeDetector, CannyDetector
import torch
from PIL import Image
import numpy as np

device = "cuda"

def get_cn_pipeleine():
    controlnets = [
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15s2_lineart_anime", torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Hosioka/Baka-DiffusionV1", controlnet=controlnets, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)    

    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, [False])
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    return pipe

def get_cn_detector(image):
    lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    canny = CannyDetector()
    lineart_anime_img = lineart_anime(image)
    canny_img = canny(image)
    canny_img = canny_img.resize((lineart_anime(image).width, lineart_anime(image).height))
    detectors = [lineart_anime_img, canny_img]
    print(detectors)
    return detectors

def generate(pipe, detectors, prompt, negative_prompt):
    default_pos = "1girl, bestquality, 4K, illustration, flatcolor, white background, no background, high contrast, "
    default_neg = "shadow, EasyNegativeV2, (worst quality, low quality:1.2), (lowres:1.2), (bad anatomy:1.2), (greyscale, monochrome:1.4), (plump, fat, curvy, belly, mature:1.4), (eyelashes, eyeshadow, mascara, lipstick:1.3), (bad-hands-5:1.2), "
    prompt = default_pos + prompt 
    negative_prompt = default_neg + negative_prompt 
    generator = torch.Generator(device)
    print(type(pipe))
    image = pipe(
                prompt=prompt,
                image=detectors,
                num_inference_steps=20,
                generator=generator,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=[1.0, 1.0],
            ).images[0]
    return image