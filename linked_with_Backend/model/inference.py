import os
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file

pipe = None  # 전역 모델 객체

def find_flux_model_attr(pipeline):
    for name, module in vars(pipeline).items():
        if module.__class__.__name__ == "FluxTransformer2DModel":
            return name, module
    raise AttributeError("FluxTransformer2DModel 컴포넌트를 찾을 수 없습니다.")

def apply_lora(module, lora_path):
    print(f"LoRA 적용 중: {lora_path}")
    state_dict = load_file(lora_path)
    module.load_state_dict(state_dict, strict=False)
    print("LoRA 적용 완료")

def load_model():
    global pipe
    print("모델 로딩 중...")
    base_model_id = "black-forest-labs/FLUX.1-dev"
    lora_path = "./downloaded_lora/pytorch_lora_weights.safetensors"

    pipe = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    attr_name, flux_model = find_flux_model_attr(pipe)
    apply_lora(flux_model, lora_path)
    setattr(pipe, attr_name, flux_model)

    print("모델과 LoRA 로딩 완료.")

def generate_images(prompt: str, negative_prompt: str = "", num_images: int = 1):
    global pipe
    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,
        guidance_scale=10,
        height=1024,
        width=1024,
        num_images_per_prompt=num_images
    )
    return results.images
