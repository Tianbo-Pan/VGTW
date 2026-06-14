---
title: VGTW Demo
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.17.1
python_version: "3.10"
app_file: demo_gradio.py
pinned: false
---

# VGTW Demo: Visual Geometry Transformer in the Wild

GitHub repository: https://github.com/Tianbo-Pan/VGTW

This repository contains the Gradio demo for **VGTW: Visual Geometry Transformer in the Wild**, a distractor-free 3D reconstruction demo built on a VGGT-style multi-view geometry backbone with LoRA adaptation and an additional predicted distractor/occlusion mask.

Given one or more input images, the demo predicts:

- camera intrinsics and extrinsics;
- depth maps and depth confidence;
- point maps and point confidence;
- a predicted distractor/occlusion mask;
- a GLB point-cloud scene for interactive visualization.

## Run the demo

```bash
pip install -r requirements.txt
python demo_gradio.py
```

The demo loads the local checkpoint by default:

```text
vgtw_lora_fp32.pt
```

For backward compatibility, `demo_gradio.py` also accepts a local `model_lora_fp32.pt`. If neither local checkpoint is present, `demo_gradio.py` falls back to downloading:

```text
pan7386/vgtw-lora/model_lora_fp32.pt
```

## Minimal code path

```python
from vgtw.models.vgtw import VGTW
from vgtw.utils.load_fn import load_and_preprocess_images

model = VGTW(lora_r=32, lora_alpha=16.0)
images = load_and_preprocess_images(["image1.jpg", "image2.jpg"])
```

The Gradio app wraps this model path and exports the reconstruction through `visual_util.predictions_to_glb`.

## Important files

```text
demo_gradio.py              # Gradio entry point and demo workflow
visual_util.py              # Converts predictions to GLB point-cloud scenes
vgtw/models/vgtw.py         # Main VGTW model class
vgtw/models/aggregator_lora.py  # LoRA-enabled multi-view token aggregator
vgtw/heads/                 # Camera, depth, point, track, and mask heads
vgtw/utils/                 # Image loading and geometry utilities
fast3r/                     # Small utility dependency used by VGTW freezing helpers
examples/                   # Optional example image sequences
skyseg.onnx                 # Optional sky filtering model
```

## Acknowledgement

This demo is based on the original VGGT codebase from Meta/Facebook Research:

```text
https://github.com/facebookresearch/vggt
```

Please also follow the original VGGT license and attribution requirements where applicable.
