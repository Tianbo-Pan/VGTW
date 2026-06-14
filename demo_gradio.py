# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import contextlib
import inspect
import torch
import numpy as np
import gradio as gr
import shutil
from datetime import datetime
import glob
import gc
import time
from huggingface_hub import hf_hub_download

from visual_util import predictions_to_glb
from vgtw.models.vgtw import VGTW
from vgtw.utils.load_fn import load_and_preprocess_images
from vgtw.utils.pose_enc import pose_encoding_to_extri_intri
from vgtw.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"


def patch_gradio_client_schema_parser():
    """
    Compatibility patch for older gradio_client versions that crash on
    JSON schema entries like {"additionalProperties": true}.
    """
    try:
        import gradio_client.utils as gr_client_utils
    except Exception:
        return

    if getattr(gr_client_utils, "_vgtw_schema_patch_applied", False):
        return

    original = gr_client_utils._json_schema_to_python_type

    def patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    gr_client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    gr_client_utils._vgtw_schema_patch_applied = True


patch_gradio_client_schema_parser()


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Initializing and loading VGTW model...")
# model = VGTW.from_pretrained("pan7386/vgtw-lora")  # another way to load the model

model = VGTW(lora_r=32, lora_alpha=16.0).to(device)
local_ckpt_path = os.path.join(os.path.dirname(__file__), "vgtw_lora_fp32.pt")
legacy_ckpt_path = os.path.join(os.path.dirname(__file__), "model_lora_fp32.pt")
if os.path.exists(local_ckpt_path):
    merged_ckpt_path = local_ckpt_path
elif os.path.exists(legacy_ckpt_path):
    merged_ckpt_path = legacy_ckpt_path
else:
    model_repo_id = os.getenv("MODEL_REPO_ID", "pan7386/vgtw-lora")
    model_filename = os.getenv("MODEL_FILENAME", "model_lora_fp32.pt")
    print(f"Local checkpoint not found. Downloading from {model_repo_id}/{model_filename}...")
    merged_ckpt_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)

state_dict = torch.load(merged_ckpt_path, map_location="cpu")
load_msg = model.load_state_dict(state_dict, strict=False)
print(f"Loaded merged checkpoint from {merged_ckpt_path}")
print(load_msg)
clear_cuda_cache()

model.eval()


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def stack_lora_predictions(raw_predictions):
    """
    Convert LoRA model outputs (list of per-view dicts) to a dict of numpy arrays.
    """
    if not isinstance(raw_predictions, list) or len(raw_predictions) == 0:
        raise ValueError("Unexpected model output format. Expected a list of per-view dicts.")

    stacked = {}
    for key in raw_predictions[0].keys():
        if key == "attn_feats":
            continue

        first_val = raw_predictions[0][key]
        if isinstance(first_val, torch.Tensor):
            stacked[key] = np.concatenate(
                [pred[key].detach().to(dtype=torch.float32).cpu().numpy() for pred in raw_predictions], axis=0
            )

    return stacked


def run_model(target_dir, model) -> dict:
    """
    Run the VGTW model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # vgtw has a known single-view output-shape issue.
    # Duplicate the only image for inference and keep the first view afterwards.
    duplicated_single_view = False
    if images.shape[0] == 1:
        duplicated_single_view = True
        images = torch.cat([images, images.clone()], dim=0)
        image_names = [image_names[0], image_names[0]]
        print("Only one image detected; duplicated to 2 views for LoRA compatibility.")

    # The LoRA model expects a list of per-view dicts instead of a raw tensor batch.
    views = []
    for idx in range(images.shape[0]):
        view = {
            "img": images[idx].unsqueeze(0),
            "file_name": [image_names[idx]],
            "img_original": [images[idx].permute(1, 2, 0) * 255],
        }
        views.append(view)

    # Run inference
    print(f"Running inference on {device}...")

    with torch.no_grad():
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            raw_predictions = model(views)

    predictions = stack_lora_predictions(raw_predictions)

    if duplicated_single_view:
        for key, value in predictions.items():
            if isinstance(value, np.ndarray) and value.shape[0] >= 2:
                predictions[key] = value[:1]

    # Fallback for checkpoints/models that do not return camera matrices directly.
    if "extrinsic" not in predictions or "intrinsic" not in predictions:
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        pose_enc = torch.from_numpy(predictions["pose_enc"]).to(device)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        predictions["extrinsic"] = extrinsic.detach().cpu().numpy()
        predictions["intrinsic"] = intrinsic.detach().cpu().numpy()

    # LoRA model predicts a mask for distractor/occlusion regions.
    # Keep it as an explicit point filter instead of modifying confidence scores.
    if "depth_mask_binary" in predictions:
        depth_mask = predictions["depth_mask_binary"]
        if depth_mask.ndim == 4 and depth_mask.shape[-1] == 1:
            depth_mask = depth_mask[..., 0]
        depth_mask = (depth_mask > 0.5).astype(np.float32)
        predictions["distractor_valid_mask"] = 1.0 - depth_mask

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    clear_cuda_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-provided
    images into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    clear_cuda_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"demo_output/input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_images:
        return None, None, None, None

    target_dir, image_paths = handle_uploads(input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Depthmap and Camera Branch",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    clear_cuda_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    clear_cuda_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer.
    """

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: loaded[key] for key in loaded.keys()}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example image sequences
# -------------------------------------------------------------------------

EXAMPLES_ROOT = "examples"


def get_example_image_sequence(example_dir):
    candidate_dirs = [example_dir, os.path.join(example_dir, "images")]
    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.webp", "*.bmp"]
    image_paths = []
    for candidate_dir in candidate_dirs:
        for pattern in image_patterns:
            image_paths.extend(glob.glob(os.path.join(candidate_dir, pattern)))
    image_paths = sorted(set(image_paths))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in example directory: {example_dir}")
    return image_paths


def discover_example_cases(examples_root):
    if not os.path.isdir(examples_root):
        return []

    discovered = []
    for case_name in sorted(os.listdir(examples_root)):
        case_dir = os.path.join(examples_root, case_name)
        if not os.path.isdir(case_dir):
            continue
        try:
            image_paths = get_example_image_sequence(case_dir)
        except ValueError:
            continue
        cover_idx = 1 if len(image_paths) > 1 else 0
        discovered.append((case_name, case_dir, image_paths[cover_idx], len(image_paths)))
    return discovered


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

custom_css = """
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
"""

blocks_kwargs = {}
if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_kwargs["theme"] = theme
if "css" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_kwargs["css"] = custom_css

with gr.Blocks(**blocks_kwargs) as demo:

    gr.HTML(
        """
    <h1> Visual Geometry Transformer in the Wild: Distractor-Free 3D Reconstruction</h1>
    <p>
    <a href="https://github.com/Tianbo-Pan/VGTW">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a set of images to create a 3D reconstruction of a scene or object. VGTW takes these images and generates a distractor-free3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use "Upload Images" on the left, or pick a quick example below to run without manual upload.</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("**Input Images**")
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            image_gallery_kwargs = {
                "label": "Preview",
                "columns": 4,
                "height": "300px",
                "object_fit": "contain",
                "preview": True,
            }
            if "buttons" in inspect.signature(gr.Gallery.__init__).parameters:
                image_gallery_kwargs["buttons"] = ["download"]

            image_gallery = gr.Gallery(**image_gallery_kwargs)

        with gr.Column(scale=5):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload images (or choose a quick example), then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Examples section ----------------------
    discovered_cases = discover_example_cases(EXAMPLES_ROOT)
    example_cards = []
    for case_name, _case_dir, cover_image_path, _image_count in discovered_cases:
        pretty_name = case_name.replace("_", " ").title()
        example_cards.append((cover_image_path, pretty_name))

    def load_example_from_gallery(
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        evt: gr.SelectData,
    ):
        case_index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        if not isinstance(case_index, int) or case_index < 0 or case_index >= len(discovered_cases):
            return (
                None,
                "Invalid example selection.",
                "None",
                gr.Dropdown(choices=["All"], value="All", interactive=True),
                None,
            )

        case_name, case_dir, _, _ = discovered_cases[case_index]
        input_images = get_example_image_sequence(case_dir)
        target_dir, image_paths = handle_uploads(input_images)
        frame_filter = "All"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        pretty_name = case_name.replace("_", " ").title()
        return glbfile, f"Loaded example: {pretty_name}. {log_msg}", target_dir, dropdown, image_paths

    if example_cards:
        gr.Markdown("Click an example image to load and reconstruct.", elem_classes=["example-log"])
        example_gallery = gr.Gallery(
            value=example_cards,
            label="Quick Examples",
            columns=4,
            height="auto",
            object_fit="cover",
            allow_preview=False,
            preview=False,
        )
        example_gallery.select(
            fn=load_example_from_gallery,
            inputs=[
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
            ],
            outputs=[
                reconstruction_output,
                log_output,
                target_dir_output,
                frame_filter,
                image_gallery,
            ],
        )
    else:
        gr.Markdown("No preload examples found under `examples/`.", elem_classes=["example-log"])

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    launch_kwargs = {"show_error": True, "share": False}
    launch_params = inspect.signature(demo.launch).parameters
    if "theme" in launch_params:
        launch_kwargs["theme"] = theme
    if "css" in launch_params:
        launch_kwargs["css"] = custom_css
    demo.queue(max_size=20).launch(**launch_kwargs)
