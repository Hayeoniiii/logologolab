import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from safetensors.torch import load_file

def find_flux_model_attr(pipeline):
    """
    FluxPipeline 내부에서 FluxTransformer2DModel 컴포넌트를 찾아
    (속성 이름, 모듈) 튜플로 반환합니다.
    """
    for name, module in vars(pipeline).items():
        if module.__class__.__name__ == "FluxTransformer2DModel":
            return name, module
    raise AttributeError("FluxTransformer2DModel 컴포넌트를 찾을 수 없습니다.")

def apply_lora(module, lora_path):
    """
    safetensors로 저장된 LoRA 가중치를 로드하여
    pipeline 컴포넌트에 덮어씁니다.
    """
    print(f"🔧 LoRA 적용 중: {lora_path}")
    state_dict = load_file(lora_path)
    module.load_state_dict(state_dict, strict=False)
    print("LoRA 적용 완료")

def main():
    # ─────────────── 설정 부분 ───────────────
    base_model_id = "black-forest-labs/FLUX.1-dev"
    lora_path     = "/workspace/retro_vintage_lora.safetensors"
    prompt        = "'logo logo lab' logo, high quality, vintage and retro style, startup company"
    output_dir    = "./outputs"
    output_name   = "retro_vintage_logo1.png"
    # ─────────────────────────────────────────

    # 1) 파이프라인 로드
    pipe = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # 2) Flux 모델 컴포넌트 찾기 및 LoRA 적용
    attr_name, flux_model = find_flux_model_attr(pipe)
    apply_lora(flux_model, lora_path)
    setattr(pipe, attr_name, flux_model)

    # 3) 추론
    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=3.5,
        height=1024,
        width=1024
    ).images[0]

    # 4) 저장
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_name)
    image.save(save_path)
    print(f"이미지 저장 완료: {save_path}")

if __name__ == "__main__":
    main()