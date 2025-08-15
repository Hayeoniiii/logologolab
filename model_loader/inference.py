import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from safetensors.torch import load_file

def find_flux_model_attr(pipeline):
    for name, module in vars(pipeline).items():
        if module.__class__.__name__ == "FluxTransformer2DModel":
            return name, module
    raise AttributeError("FluxTransformer2DModel 컴포넌트를 찾을 수 없습니다.")

def apply_lora(module, lora_path): #LoRA 가중치 load
    print(f"LoRA 적용 중: {lora_path}")
    state_dict = load_file(lora_path)
    module.load_state_dict(state_dict, strict=False)
    print("LoRA 적용 완료")

def main():
    base_model_id = "black-forest-labs/FLUX.1-dev"
    lora_path     = "/workspace/downloaded_lora/pytorch_lora_weights.safetensors"
    prompt        = "cute playful logo, colorful cartoon mascot, rounded sans-serif font, cheerful design, icon with text, featuring the text 'Logologo Lab'"
    output_dir    = "./outputs"
    num_images_per_prompt = 2  # 생성할 이미지 개수
    
    # 1) 파이프라인 로드
    pipe = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # 2) LoRA 적용
    attr_name, flux_model = find_flux_model_attr(pipe)
    apply_lora(flux_model, lora_path)
    setattr(pipe, attr_name, flux_model)

    # 3) 추론 (한 번에 num_images_per_prompt 장 생성)
    results = pipe(
        prompt=prompt,
        num_inference_steps=100,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        num_images_per_prompt=num_images_per_prompt
    )
    images = results.images

    # 4) 저장
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(images, start=1):
        save_name = f"{idx}번.png"
        save_path = os.path.join(output_dir, save_name)
        img.save(save_path)
        print(f"이미지 저장 완료: {save_path}")

if __name__ == "__main__":
    main()