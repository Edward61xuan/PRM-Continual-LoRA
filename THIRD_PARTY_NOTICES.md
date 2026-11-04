# Third-party notices

Original PRM code is licensed under the root MIT LICENSE. The following
adapted third-party code retains its upstream license.

## timm Vision Transformer

- Local file: `prm/vit.py`
- Source: [`timm/models/vision_transformer.py`, timm v0.9.12](https://github.com/huggingface/pytorch-image-models/blob/v0.9.12/timm/models/vision_transformer.py)
- Upstream attribution: "Hacked together by / Copyright 2020, Ross Wightman"
- License: Apache License, Version 2.0. The complete upstream license is
  included in [licenses/timm-APACHE-2.0.txt](licenses/timm-APACHE-2.0.txt).
- PRM modifications: trimmed model components and factories; propagated a
  `perturbation` argument through the transformer blocks and attention to the
  K/V LoRA wrapper; added task-specific head/text-feature handling and
  CLIP-compatible logit output; adapted model registration and checkpoint
  handling for the retained factories.

The upstream file acknowledges the paper authors and their released code and
weights, Phil Wang's vit-pytorch class-token implementation, Andrej Karpathy's
minGPT transformer style, and the Hugging Face Transformers and TensorFlow
BERT reference implementations. Those acknowledgments are retained in
`prm/vit.py`.
