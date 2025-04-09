# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import torch

from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
from modeling import build_model
from modeling.BaseModel import BaseModel
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from utilities.distributed import init_distributed


zero_tensor = torch.zeros(1, 1, 1)


@dataclass
class PredictionTarget:
    target: str
    pred_mask: torch.Tensor = zero_tensor
    adjusted_p_value: float = -1.0


class Model:
    def init(self):
        self._model = init_model()

    def predict(
        self, image: Image.Image, modality_type: str, targets: list[str]
    ) -> Tuple[Image.Image, str]:
        image_annotated, prediction_targets_not_found = predict(
            self._model, image, modality_type, targets
        )
        targets_not_found_str = (
            "\n".join(
                f"{t.target} ({t.adjusted_p_value:.3f})"
                for t in prediction_targets_not_found
            )
            if prediction_targets_not_found
            else "All targets were found!"
        )
        return image_annotated, targets_not_found_str


def init_model() -> BaseModel:
    # --- Start Modification ---
    # Read the token from the environment variable
    hf_token = os.getenv('HF_TOKEN')
    if hf_token is None:
        print("Warning: HF_TOKEN environment variable not set. Download might fail for gated models.")

    # Download model, explicitly passing the token
    model_file = hf_hub_download(
        repo_id="microsoft/BiomedParse",
        filename="biomedparse_v1.pt",
        token=hf_token  # <--- Pass the token here
    )
    # --- End Modification ---

    # Initialize model (rest of the function is unchanged)
    conf_files = "configs/biomedparse_inference.yaml"
    opt = load_opt_from_config_files([conf_files])
    opt = init_distributed(opt)

    model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    return model


def predict(
    model: BaseModel, image: Image.Image, modality_type: str, targets: list[str]
) -> Tuple[Image.Image, list[PredictionTarget]]:
    assert len(targets) > 0, "At least one target is required"

    prediction_tasks = [PredictionTarget(target=target) for target in targets]

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get predictions
    pred_mask = interactive_infer_image(model, image, targets)

    for i, pt in enumerate(prediction_tasks):
        pt.pred_mask = pred_mask[i]

    image_np = np.array(image)

    for pt in prediction_tasks:
        adj_p_value = check_mask_stats(
            image_np, pt.pred_mask * 255, modality_type, pt.target
        )
        pt.adjusted_p_value = float(adj_p_value)

    pred_targets_found, pred_tasks_not_found = segregate_prediction_tasks(
        prediction_tasks, 0.05
    )

    # Generate visualization
    colors = generate_colors(len(pred_targets_found))
    masks = [1 * (pred_mask[i] > 0.5) for i in range(len(pred_targets_found))]
    pred_overlay = overlay_masks(image, masks, colors)

    pred_overlay = add_legend(pred_overlay, pred_targets_found, colors)

    return pred_overlay, pred_tasks_not_found


def segregate_prediction_tasks(
    prediction_tasks: list[PredictionTarget], p_value_threshold: float
) -> tuple[list[PredictionTarget], list[PredictionTarget]]:
    """Segregates Prediction Tasks by p-value

    Prediction tasks with a p-value higher than p_value_threshold go into the targets_found list.
    Otherwise, they go into the targets_not_found list.
    """

    targets_found = []
    targets_not_found = []
    for pt in prediction_tasks:
        if pt.adjusted_p_value > p_value_threshold:
            targets_found.append(pt)
        else:
            targets_not_found.append(pt)

    return targets_found, targets_not_found


def generate_colors(n: int) -> list[Tuple[int, int, int]]:
    cmap = plt.get_cmap("tab10")
    colors = [
        (int(255 * cmap(i)[0]), int(255 * cmap(i)[1]), int(255 * cmap(i)[2]))
        for i in range(n)
    ]
    return colors


def overlay_masks(
    image: Image.Image,
    masks: list[np.ndarray],
    colors: list[Tuple[int, int, int]],
) -> Image.Image:
    overlay = image.copy()
    overlay = np.array(overlay, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(
            np.uint8
        )
    return Image.fromarray(overlay)


def add_legend(
    image: Image.Image,
    pred_targets_found: list[PredictionTarget],
    colors: list[Tuple[int, int, int]],
) -> Image.Image:
    if len(pred_targets_found) == 0:
        return image

    # Convert to numpy for manipulation
    pred_overlay = np.array(image)

    # Calculate dimensions based on image resolution
    image_width = pred_overlay.shape[1]
    font_size = max(16, int(image_width * 0.02))  # Scale with image width, minimum 16px
    box_size = int(font_size * 1.5)  # Color box proportional to font
    entry_height = int(box_size * 1.5)  # Space between entries
    legend_padding = int(font_size * 0.75)  # Padding scales with font

    # Calculate total legend height
    legend_height = entry_height * len(pred_targets_found)
    total_height = pred_overlay.shape[0] + legend_height + 2 * legend_padding

    # Create new image with space for legend
    new_image = np.zeros((total_height, pred_overlay.shape[1], 3), dtype=np.uint8)
    new_image[: pred_overlay.shape[0], :] = pred_overlay
    new_image[pred_overlay.shape[0] :] = 255  # White background for legend

    # Convert to PIL once for all legend entries
    img_pil = Image.fromarray(new_image)
    draw = ImageDraw.Draw(img_pil)

    # Try to load a system font with proper scaling
    font = None
    system_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]
    for font_path in system_fonts:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        # Fallback to default font if no system fonts are available
        font = ImageFont.load_default()

    # Get font metrics for proper vertical centering
    bbox = font.getbbox("Aj")  # Use tall characters to get true height
    font_height = bbox[3] - bbox[1]  # bottom - top

    # Draw legend entries
    start_y = pred_overlay.shape[0] + legend_padding
    for i, task in enumerate(pred_targets_found):
        # Draw color box
        box_x = legend_padding
        box_y = start_y + i * entry_height
        box_coords = (box_x, box_y, box_x + box_size, box_y + box_size)
        draw.rectangle(box_coords, fill=colors[i])

        # Draw text (vertically centered with color box)
        text_y = box_y + (box_size - font_height) // 2  # Center text with box
        # Format text with truncated p-value
        p_value_truncated = "{:.2f}".format(task.adjusted_p_value)
        legend_text = f"{task.target} ({p_value_truncated})"
        draw.text(
            (box_x + box_size + legend_padding, text_y),
            legend_text,
            fill=(0, 0, 0),
            font=font,
        )

    return img_pil
