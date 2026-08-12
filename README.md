# 🎥 Lightweight Video Segmentation

### EfficientTAM · nanoVLM · SmolVLM

A lightweight computer vision project for **frame-based video segmentation** using modern vision and vision-language models.

The project processes video sequences as **pre-extracted image frames**, applies the implemented models to the frames, and generates segmentation masks and processed outputs.

---

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github\&logoColor=white)](https://github.com/bamimahyar/video-segmentation-lightweight)

</p>

---

## ✨ Overview

Video segmentation is the task of identifying and segmenting objects across a sequence of video frames.

Large vision models can provide strong performance, but they may require significant computational resources. This project investigates a more lightweight approach using modern efficient models and a **frame-based processing pipeline**.

Instead of requiring the original video file during inference, the project works with video data that has already been converted into individual image frames.

### Pipeline

```text
                    VIDEO DATA
                        │
                        ▼
               Pre-extracted Frames
                        │
                        ▼
              ┌───────────────────┐
              │   Model Inference │
              └───────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     EfficientTAM    nanoVLM      SmolVLM
          │             │             │
          ▼             ▼             ▼
     Segmentation    Vision-       Vision-
        Masks        Language      Language
                     Analysis      Analysis
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                  Output Frames
```

---

## 🚀 Key Features

* 🎯 Frame-based video segmentation
* ⚡ Lightweight model experimentation
* 🧠 EfficientTAM for segmentation
* 👁️ nanoVLM for lightweight vision-language processing
* 🔬 SmolVLM for compact multimodal processing
* 🖼️ Support for individual image-frame inference
* 📊 IoU-based segmentation evaluation
* 🧩 Modular project structure
* 📝 Reproducible configuration and model comparison

---

# 🤖 Models

## EfficientTAM

**EfficientTAM** is the primary segmentation model used in this project.

It is designed for efficient video object segmentation and tracking with a focus on reducing computational and memory requirements.

In this project, video sequences are represented as ordered image frames. EfficientTAM processes these frames and generates the corresponding segmentation masks.

🔗 **Official Repository**

https://github.com/yformer/EfficientTAM

---

## nanoVLM

**nanoVLM** is a lightweight Vision-Language Model developed by Hugging Face.

It is included in this project for lightweight multimodal experimentation and model comparison.

🔗 **Official Repository**

https://github.com/huggingface/nanoVLM

🔗 **Official Introduction**

https://huggingface.co/blog/nanovlm

> **Note:** nanoVLM is a Vision-Language Model and is not treated as the primary segmentation model in this project.

---

## SmolVLM

**SmolVLM** is a compact Vision-Language Model developed by Hugging Face.

It is included for lightweight multimodal processing and comparison with other implemented models.

🔗 **Documentation**

https://huggingface.co/docs/transformers/model_doc/smolvlm

🔗 **Official Introduction**

https://huggingface.co/blog/smolvlm

> **Note:** SmolVLM is a Vision-Language Model and is not presented as the primary segmentation model.

---

# 📂 Dataset

The dataset used in this project consists of **video sequences represented as pre-extracted image frames**.

The original videos are converted into ordered frames before the segmentation stage.

Example structure:

```text
dataset/
│
├── Video_01/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── 00004.jpg
│   └── ...
│
├── Video_02/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   └── ...
│
└── ...
```

This design allows each video sequence to be processed frame by frame while preserving the temporal order of the original video.

---

# 🔬 Processing Workflow

The segmentation workflow can be summarized as follows:

```text
┌──────────────────────────┐
│   Pre-extracted Frames   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Input Frame         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     Model Inference      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Segmentation Mask     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Processed Frame       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Output Results      │
└──────────────────────────┘
```

For a complete video sequence:

```text
Frame 001 ──► Model ──► Mask 001
Frame 002 ──► Model ──► Mask 002
Frame 003 ──► Model ──► Mask 003
    ⋮
Frame N   ──► Model ──► Mask N
```

The processed frames can subsequently be combined or visualized as a segmented video sequence.

---

# 📁 Project Structure

```text
video-segmentation-lightweight/
│
├── models/
│   ├── efficienttam/
│   ├── nanovlm/
│   └── smolvlm/
│
├── utils/
│
├── outputs/
│
├── sample_input/
│
├── configs/
│
├── main.py
├── process_all_models.py
├── requirements.txt
└── README.md
```

### Directory Description

| Directory / File        | Description                           |
| ----------------------- | ------------------------------------- |
| `models/`               | Model implementations                 |
| `utils/`                | Helper functions and utilities        |
| `outputs/`              | Generated masks and processed results |
| `sample_input/`         | Sample input frames                   |
| `configs/`              | Model and project configurations      |
| `main.py`               | Main project entry point              |
| `process_all_models.py` | Runs the implemented models           |
| `requirements.txt`      | Python dependencies                   |
| `README.md`             | Project documentation                 |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/bamimahyar/video-segmentation-lightweight.git
cd video-segmentation-lightweight
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. GPU Support

For GPU acceleration, make sure that your installed PyTorch version is compatible with your system's CUDA version.

GPU execution is recommended for faster model inference.

---

# ▶️ Usage

## Process a Single Frame

To process an individual input frame:

```bash
python process_all_models.py --image sample_input/00035.jpg
```

The available models process the input according to their implementation and generate the corresponding outputs.

---

## Process a Video Sequence

The project uses **pre-extracted video frames** rather than requiring the original video file during the segmentation stage.

The workflow is:

```text
Video
  │
  ▼
Pre-extracted Frames
  │
  ├── Frame 001
  ├── Frame 002
  ├── Frame 003
  └── ...
        │
        ▼
   Model Processing
        │
        ▼
 Segmentation Results
        │
        ▼
 Processed Frames
```

This approach is particularly useful when working with existing frame-based datasets.

---

# 📤 Outputs

Generated results are stored in:

```text
outputs/
```

Depending on the selected model and configuration, the output directory may contain:

```text
outputs/
│
├── segmentation_masks/
├── processed_frames/
└── model_results/
```

The exact output structure depends on the current model implementation.

---

# 📊 Evaluation

The primary segmentation metric used for quantitative evaluation is:

## Intersection over Union (IoU)

IoU measures the overlap between a predicted segmentation mask and its corresponding ground-truth mask.

```text
IoU = Area of Intersection
      ───────────────────────
        Area of Union
```

Higher IoU values indicate greater overlap between the prediction and ground truth.

### Evaluation Requirements

For a reproducible evaluation, the following are required:

* Predicted segmentation masks
* Corresponding ground-truth masks
* Consistent image/frame ordering
* Clearly defined evaluation procedure

---

# 🧪 Experimental Results

Quantitative results are intentionally reported only after running a reproducible evaluation against the corresponding ground-truth masks.

| Model        | IoU | Inference Time | FPS |
| ------------ | --: | -------------: | --: |
| EfficientTAM | TBD |            TBD | TBD |
| nanoVLM      | TBD |            TBD | TBD |
| SmolVLM      | TBD |            TBD | TBD |

> **TBD** values should be replaced with results obtained from actual experiments.

No unverified IoU value is reported as the final project result.

---

# ⚠️ Limitations

* The dataset is provided as pre-extracted image frames.
* The segmentation stage operates on individual frames rather than directly reading MP4 files.
* EfficientTAM is the primary segmentation-oriented model in the project.
* nanoVLM and SmolVLM are primarily Vision-Language Models and are included for multimodal experimentation and comparison.
* Model performance depends on the dataset, model configuration, hardware, and evaluation methodology.
* Quantitative results should be reported only after evaluation against appropriate ground-truth masks.

---

# 🔮 Future Improvements

Potential future improvements include:

* Automated video-to-frame extraction
* Automatic frame-to-video reconstruction
* More comprehensive quantitative evaluation
* Mean IoU across complete video sequences
* FPS and inference-time benchmarking
* GPU memory benchmarking
* Additional lightweight segmentation models
* Improved visualization of segmentation masks
* Automated experiment tracking

---

# 📚 References

## EfficientTAM

Yang, F. et al. **EfficientTAM: Fast and Memory-Efficient Tracking and Segmentation.**

Official repository:

https://github.com/yformer/EfficientTAM

---

## nanoVLM

Hugging Face. **nanoVLM: A Tiny Vision-Language Model.**

Official repository:

https://github.com/huggingface/nanoVLM

Official introduction:

https://huggingface.co/blog/nanovlm

---

## SmolVLM

Hugging Face. **SmolVLM: compact vision-language models.**

Official documentation:

https://huggingface.co/docs/transformers/model_doc/smolvlm

Official introduction:

https://huggingface.co/blog/smolvlm

---

# 📖 Citation

If this project is used in academic research, please cite the original papers and official repositories of the models used in the implementation.

---

# 👨‍💻 Author

**Mahyar Bami**

GitHub:

https://github.com/bamimahyar

Project:

https://github.com/bamimahyar/video-segmentation-lightweight

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

<p align="center">

**Lightweight Models · Video Frames · Segmentation · Computer Vision**

</p>
