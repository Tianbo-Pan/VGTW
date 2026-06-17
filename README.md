<div align="center">
<h1>VGTW</h1>
<h3>Visual Geometry Transformer in the Wild</h3>

<a href="https://tianbo-pan.github.io/vgt-w/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://github.com/Tianbo-Pan/VGTW" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub repository"></a>
<a href="https://huggingface.co/pan7386/vgtw-lora/blob/main/vgtw_lora_fp32.pt" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-vgtw--lora-blue" alt="VGTW LoRA checkpoint"></a>
<a href="https://github.com/facebookresearch/vggt" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Based%20on-VGGT-green" alt="Based on VGGT"></a>
<a href="./LICENSE.txt"><img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey" alt="License"></a>

<p>
  <span class="author"><a href="https://github.com/Tianbo-Pan">Tianbo Pan</a></span>
</p>

<strong>Distractor-free 3D reconstruction from unconstrained image collections.</strong>
</div>

## Overview

VGTW is a compact research/demo repository for reconstructing 3D geometry in the wild, where input views may contain distractors, occluders, sky, or inconsistent backgrounds. It builds on the VGGT visual-geometry backbone and adds a LoRA-adapted VGTW head that predicts geometry together with a distractor/occlusion mask for cleaner point-cloud export.

Given one or more images, VGTW predicts camera parameters, dense depth, world points, confidence maps, and a distractor-aware point filter. The Gradio demo turns these predictions into an interactive GLB scene that can be viewed directly in the browser or downloaded for later use.

## Highlights

- **VGGT-style feed-forward geometry:** predicts cameras, depth, and dense world points from image sequences.
- **In-the-wild filtering:** estimates a binary distractor/occlusion mask and uses it to suppress unreliable points.
- **LoRA checkpoint:** lightweight adaptation distributed as `vgtw_lora_fp32.pt`.
- **Browser demo:** upload images, tune confidence/background filters, and export a GLB point-cloud scene.
- **Built-in examples:** sample image sets are provided under [`examples/`](./examples) for quick testing.

## Pretrained checkpoint

The demo automatically uses the following checkpoint. If it is not found locally, `demo_gradio.py` downloads it from Hugging Face.

| Model | Resolution | Adaptation | Download |
| :--- | :---: | :---: | :--- |
| `VGTW-LoRA` | 518 px width | LoRA | [Hugging Face](https://huggingface.co/pan7386/vgtw-lora/blob/main/vgtw_lora_fp32.pt) |

To keep a local copy, place the file at the repository root:

```text
vgtw_lora_fp32.pt
```

For backward compatibility, the demo also checks for `model_lora_fp32.pt`.

## Quick start

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Tianbo-Pan/VGTW.git
cd VGTW
pip install -r requirements.txt
```

Run the model from Python:

```python
import torch

from vgtw.models.vgtw import VGTW
from vgtw.utils.load_fn import load_and_preprocess_images

checkpoint_path = "vgtw_lora_fp32.pt"
image_names = ["path/to/imageA.jpg", "path/to/imageB.jpg", "path/to/imageC.jpg"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGTW(lora_r=32, lora_alpha=16.0).to(device).eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

images = load_and_preprocess_images(image_names).to(device)
views = [
    {
        "img": images[i].unsqueeze(0),
        "file_name": [image_names[i]],
        "img_original": [images[i].permute(1, 2, 0) * 255],
    }
    for i in range(images.shape[0])
]

with torch.inference_mode():
    predictions = model(views)

# Each item in `predictions` corresponds to one input view and includes
# depth, depth_conf, world_points, camera matrices, and distractor masks.
```

## Interactive demo

Launch the local Gradio app:

```bash
python demo_gradio.py
```

The interface supports uploaded images and bundled examples, then lets you:

- reconstruct a distractor-free point cloud;
- choose depth-based or pointmap-based visualization;
- adjust confidence filtering;
- optionally filter sky, black backgrounds, or white backgrounds;
- inspect and download the exported GLB scene.

## Output dictionary

The VGTW forward pass returns a list of per-view dictionaries. Common keys include:

| Key | Description |
| :--- | :--- |
| `extrinsic`, `intrinsic` | predicted camera matrices |
| `depth`, `depth_conf` | dense depth and depth confidence |
| `world_points`, `world_points_conf` | dense 3D points and confidence |
| `depth_mask_binary` | predicted distractor/occlusion mask |
| `refined_mask_binary` | mask used by the visualization pipeline |
| `images` | preprocessed input images |

## Repository layout

```text
VGTW/
├── demo_gradio.py          # Gradio reconstruction demo
├── visual_util.py          # GLB/point-cloud visualization utilities
├── vgtw/                   # VGTW model, heads, layers, and geometry utils
├── fast3r/                 # imported support modules
├── examples/               # sample image collections
├── requirements.txt
└── requirements_demo.txt
```

## Acknowledgements

This repository is derived from and builds upon the original [VGGT](https://github.com/facebookresearch/vggt) codebase from Meta/Facebook Research. Please follow the upstream license and attribution requirements where applicable.

## License

See [`LICENSE.txt`](./LICENSE.txt) for the license terms of this repository.
