import os
import copy
import torch
from pathlib import Path
from diffusers import FluxPipeline
from safetensors.torch import load_file

pipe = None  # 전역 모델 객체
_base_transformer_state = None  # 원본 가중치 백업
_current_style = None  # 현재 적용된 스타일

# 스타일 → LoRA 경로 매핑
LORA_PATHS = {
    "simple": "./loras/simple_logo.safetensors",
    "minimal": "./loras/simple_logo.safetensors",
    "retro": "./loras/vintage_logo.safetensors",
    "vintage": "./loras/vintage_logo.safetensors",
    "cute": "./loras/cute_logo.safetensors",
    "playful": "./loras/cute_logo.safetensors",
    "luxury": "./loras/luxury_logo.safetensors",
    "tattoo": "./loras/tattoo_logo.safetensors",
    "futuristic": "./loras/futuristic_logo.safetensors",
    "cartoon": "./loras/cartoon_logo.safetensors",
    "watercolor": "./loras/watercolor_logo.safetensors",
    "none": None, 
}

def _find_flux_transformer(pipeline: FluxPipeline):
    # FluxPipeline 내부에서 FluxTransformer2DModel을 찾는다.
    for name, module in vars(pipeline).items():
        if getattr(module, "__class__", None) and module.__class__.__name__ == "FluxTransformer2DModel":
            return name, module
    raise AttributeError("FluxTransformer2DModel 컴포넌트를 찾을 수 없습니다.")

def _apply_lora_to_module(module, lora_path: str):
    state_dict = load_file(lora_path)
    module.load_state_dict(state_dict, strict=False)

def _reset_transformer_to_base():
    global pipe, _base_transformer_state
    _, transformer = _find_flux_transformer(pipe)
    transformer.load_state_dict(_base_transformer_state, strict=True)

def load_model():
    global pipe, _base_transformer_state, _current_style
    if pipe is not None:
        return

    print("FLUX 모델 로딩 중...")
    base_model_id = "black-forest-labs/FLUX.1-dev"

    pipe = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # 원본 transformer 가중치 백업
    _, transformer = _find_flux_transformer(pipe)
    # clone()으로 CPU에 안전 백업 (메모리 여유 필요)
    _base_transformer_state = {k: v.detach().cpu().clone() for k, v in transformer.state_dict().items()}
    _current_style = "none"
    print("베이스 모델 로딩 및 원본 가중치 백업 완료.")

def _ensure_style(style: str):
   
    global _current_style, pipe

    if pipe is None:
        load_model()

    style = (style or "none").strip().lower()
    if style not in LORA_PATHS:
        print(f"알 수 없는 style '{style}', LoRA 미적용으로 진행합니다.")
        style = "none"

    if style == _current_style:
        return  

    # 항상 원본으로 초기화 후 LoRA 재적용
    _reset_transformer_to_base()

    lora_path = LORA_PATHS[style]
    if lora_path:
        path = Path(lora_path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA 파일을 찾을 수 없습니다: {path}")
        print(f"LoRA 적용: {path}")
        _, transformer = _find_flux_transformer(pipe)
        _apply_lora_to_module(transformer, str(path))
        print("LoRA 적용 완료.")
    else:
        print("LoRA 미적용")

    _current_style = style

@torch.inference_mode()
def generate_images(
    prompt: str,
    style: str,
    negative_prompt: str = "no watermark, captions, extra words, low quality",
    num_images: int = 1,
):

    global pipe
    if pipe is None:
        load_model()

    _ensure_style(style)

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
