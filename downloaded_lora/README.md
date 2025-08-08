---
base_model: black-forest-labs/FLUX.1-dev
library_name: diffusers
license: other
instance_prompt: retro vintage logo, distressed texture, old-school serif font, classic
  badge design
widget: []
tags:
- text-to-image
- diffusers-training
- diffusers
- lora
- flux
- flux-diffusers
- template:sd-lora
---

<!-- This model card has been generated automatically according to the information the training script had access to. You
should probably proofread and complete it, then remove this comment. -->


# Flux DreamBooth LoRA - logologolab/retro_vintage_logo_lora

<Gallery />

## Model description

These are logologolab/retro_vintage_logo_lora DreamBooth LoRA weights for black-forest-labs/FLUX.1-dev.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? False.

## Trigger words

You should use `retro vintage logo, distressed texture, old-school serif font, classic badge design` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA](logologolab/retro_vintage_logo_lora/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('logologolab/retro_vintage_logo_lora', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('retro vintage logo, distressed texture, old-school serif font, classic badge design').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).


## Intended uses & limitations

#### How to use

```python
# TODO: add an example code snippet for running this diffusion pipeline
```

#### Limitations and bias

[TODO: provide examples of latent issues and potential remediations]

## Training details

[TODO: describe the data used to train the model]