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

import json
import os
from pathlib import Path
from typing import Dict, List

from inference_utils.target_dist import modality_targets_from_target_dist

# If True, then mock init_model() and predict() functions will be used.
DEV_MODE = True if os.getenv("DEV_MODE") else False

import gradio as gr

if DEV_MODE:
    from inference_utils.model_mock import Model
else:
    from inference_utils.model import Model


gr.set_static_paths(["assets"])


description = """\
This Space is based on the [BiomedParse model](https://microsoft.github.io/BiomedParse/).

BiomedParse is a model that can detect various targets like organs, diseases, and more
in biomedical images. The biomedical images can be of different types like CT, MRI, X-ray, etc.

> Note: Don't use this model for medical diagnosis. Always consult a healthcare professional for medical advice.

## How to use this demo

1. Upload a biomedical image
2. Select the modality type
3. Select the targets you want to detect
4. Click on the 'Submit' button to see the prediction

The model will highlight the detected targets in the image and show the targets that were not found below the image.
Each found target is represented by a different color.
Each target comes with a p-value that is computed using the Kolmogorov-Smirnov test.
A target whose p-value is below 0.05 is considered "not found".
For more details, check out the paper [BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once](https://arxiv.org/abs/2405.12971).

## Modality Types

- [Computed Tomography (CT)](https://en.wikipedia.org/wiki/Computed_tomography)
- [Magnetic Resonance Imaging (MRI)](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging)
- [X-ray](https://en.wikipedia.org/wiki/X-ray)
- [Medical Ultrasound](https://en.wikipedia.org/wiki/Medical_ultrasound)
- [Pathology](https://en.wikipedia.org/wiki/Pathology)
- [Fundus (eye)](https://en.wikipedia.org/wiki/Fundus_(eye))
- [Dermoscopy](https://en.wikipedia.org/wiki/Dermoscopy)
- [Endoscopy](https://en.wikipedia.org/wiki/Endoscopy)
- [Optical Coherence Tomography (OCT)](https://en.wikipedia.org/wiki/Optical_coherence_tomography)

"""


examples = [
    ["examples/144DME_as_F.jpeg", "OCT", []],
    ["examples/C3_EndoCV2021_00462.jpg", "Endoscopy", []],
    ["examples/CT-abdomen.png", "CT-Abdomen", []],
    ["examples/covid_1585.png", "X-Ray-Chest", []],
    ["examples/ISIC_0015551.jpg", "Dermoscopy", []],
    [
        "examples/LIDC-IDRI-0140_143_280_CT_lung.png",
        "CT-Chest",
        [],
    ],
    [
        "examples/Part_1_516_pathology_breast.png",
        "Pathology",
        [],
    ],
    ["examples/T0011.jpg", "Fundus", []],
    [
        "examples/TCGA_HT_7856_19950831_8_MRI-FLAIR_brain.png",
        "MRI-FLAIR-Brain",
        [],
    ],
]


def load_modality_targets() -> Dict[str, List[str]]:
    target_dist_json_path = Path("inference_utils/target_dist.json")
    with open(target_dist_json_path, "r") as f:
        target_dist = json.load(f)

    modality_targets = modality_targets_from_target_dist(target_dist)
    print(f"DEBUG: MODALITY_TARGETS = {modality_targets}")  # Add this line
    return modality_targets


MODALITY_TARGETS = load_modality_targets()
DEFAULT_MODALITY = "CT-Abdomen"


def run():
    model = Model()
    model.init()

    with gr.Blocks() as demo:
        gr.Markdown("# BiomedParse Demo")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                input_modality_type = gr.Dropdown(
                    choices=list(MODALITY_TARGETS.keys()),
                    label="Modality Type",
                    value=DEFAULT_MODALITY,
                )
                input_targets = gr.CheckboxGroup(
                    choices=MODALITY_TARGETS[DEFAULT_MODALITY],
                    label="Targets",
                )
            with gr.Column():
                output_image = gr.Image(type="pil", label="Prediction")
                output_targets_not_found = gr.Textbox(
                    label="Targets Not Found", lines=4, max_lines=10
                )

        input_modality_type.change(
            fn=update_input_targets,
            inputs=input_modality_type,
            outputs=input_targets,
        )

        submit_btn = gr.Button("Submit")
        submit_btn.click(
            fn=model.predict,
            inputs=[input_image, input_modality_type, input_targets],
            outputs=[output_image, output_targets_not_found],
        )

        gr.Examples(
            examples=examples,
            inputs=[input_image, input_modality_type, input_targets],
            outputs=[output_image, output_targets_not_found],
            fn=model.predict,
            cache_examples=False,
        )

    return demo

def update_input_targets(input_modality_type):
    print(f"DEBUG: input_modality_type = {input_modality_type}")  # Add this line
    choices = MODALITY_TARGETS[input_modality_type]
    print(f"DEBUG: choices = {choices}") # Add this line
    return gr.CheckboxGroup(
        choices=choices,
        value=[],
        label="Targets",
    )


demo = run()

if __name__ == "__main__":
    print("--- Entered __main__ block ---")
    # Assuming run() defines and returns the demo object internally
    # and doesn't launch it. We call launch on the returned object.
    demo = run() # Keep this if run() defines the Blocks but doesn't launch
    print(f"--- run() completed, demo object type: {type(demo)} ---")
    print("--- Calling demo.launch(server_name='0.0.0.0', server_port=7860) ---")
    # Make sure demo is a Gradio Blocks or Interface object here
    if hasattr(demo, 'launch'):
         demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
         print("--- demo.launch() returned (should not happen if running normally) ---")
    else:
         print("--- ERROR: demo object does not have launch method ---")
