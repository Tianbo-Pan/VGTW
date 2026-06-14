<div align="center">

# VGTW: Visual Geometry Transformer in the Wild

**Distractor-Free 3D Reconstruction from Unconstrained Images**

<a href="https://huggingface.co/pan7386/vgtw-lora/blob/main/vgtw_lora_fp32.pt">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-vgtw--lora-blue" alt="VGTW LoRA Checkpoint">
</a>
<a href="https://github.com/facebookresearch/vggt">
  <img src="https://img.shields.io/badge/Based%20on-VGGT-green" alt="Based on VGGT">
</a>

</div>

## Overview

VGTW is a Gradio demo for distractor-free 3D reconstruction in the wild. It builds on a VGGT-style multi-view geometry backbone and adds LoRA adaptation plus a predicted distractor/occlusion mask.

Given one or more input images, the demo estimates camera parameters, dense geometry, confidence maps, and a distractor-aware point filter, then exports an interactive GLB point-cloud scene.

## Checkpoint

The demo uses the following checkpoint by default:

| Model | File | Download |
|:--|:--|:--|
| VGTW LoRA | `vgtw_lora_fp32.pt` | [Hugging Face](https://huggingface.co/pan7386/vgtw-lora/blob/main/vgtw_lora_fp32.pt) |

Place the checkpoint in the repository root:

```text
vgtw_lora_fp32.pt
```

For backward compatibility, the demo also accepts a local `model_lora_fp32.pt`. If neither local checkpoint exists, it downloads `vgtw_lora_fp32.pt` from `pan7386/vgtw-lora` automatically.

## Installation

```bash
git clone https://github.com/Tianbo-Pan/VGTW.git
cd VGTW
pip install -r requirements.txt
```

## Run the Gradio Demo

```bash
python demo_gradio.py
```

The interface lets you:

- upload image sequences or select built-in examples;
- reconstruct a distractor-free point cloud;
- switch between depth-based and pointmap-based visualization;
- adjust confidence filtering;
- optionally filter sky, black background, or white background;
- view and download the exported GLB scene.

## Minimal Usage

```python
from vgtw.models.vgtw import VGTW
from vgtw.utils.load_fn import load_and_preprocess_images

model = VGTW(lora_r=32, lora_alpha=16.0)
images = load_and_preprocess_images(["image1.jpg", "image2.jpg"])
```


## Acknowledgement

This demo is derived from the original VGGT codebase from Meta/Facebook Research:

```text
https://github.com/facebookresearch/vggt
```

Please follow the original VGGT license and attribution requirements where applicable.
