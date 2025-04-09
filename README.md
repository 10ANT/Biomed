---
title: Biomedparse Docker
emoji: 📉
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
---

# BiomedParse Hugging Face Space

This Hugging Face Space provides an interactive Gradio interface to explore the functionality of **BiomedParse**. BiomedParse is a foundation model for joint segmentation, detection, and recognition of biomedical objects across nine modalities.

This Space allows you to:

- Upload biomedical data.
- Use the BiomedParse model for analysis.
- View and interact with the results.

## Acknowledgments

This Space is based on the work by the research team behind BiomedParse.

- GitHub Repository: [BiomedParse GitHub Repository](https://github.com/microsoft/BiomedParse)
- Hugging Face Model: [BiomedParse Model](https://huggingface.co/microsoft/BiomedParse)

All rights to the model and its underlying research are held by the original authors. This Space only provides an interface for interacting with their published model.

## How It Works

- This Space leverages the [BiomedParse model](https://huggingface.co/microsoft/BiomedParse) hosted on Hugging Face.
- The Space fetches the model directly from Hugging Face each time it is run.
- The Python backend is partially adapted from the original BiomedParse GitHub repository under the [Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html).

## Licensing

- **Code in this Space**: Licensed under the [Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html), as per the original BiomedParse GitHub repository.
- **Model**: The BiomedParse model is licensed under the [Creative Commons Attribution Non Commercial Share Alike 4.0](https://spdx.org/licenses/CC-BY-NC-SA-4.0.html). Ensure that your use complies with the terms of this license.

## Development of the Gradio interface

To develop the Gradio interface locally against a mock ML model, execute

    make run
