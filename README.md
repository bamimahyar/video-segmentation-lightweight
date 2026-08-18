🎥 Lightweight Video Segmentation

EfficientTAM · nanoVLM · SmolVLM

Repository: https://github.com/bamimahyar/video-segmentation-lightweight

A lightweight computer vision project for frame-based video segmentation using modern vision and vision-language models.

This project processes video sequences as pre-extracted image frames, applies the implemented models to those frames, and generates segmentation masks and processed outputs.

⸻

✨ Overview

Video segmentation is the task of identifying and segmenting objects across a sequence of video frames.

Instead of requiring the original video file during the segmentation stage, this project works with video data that has already been converted into individual image frames.

Processing Overview

                        Video Data
                            │
                            ▼
                   Pre-extracted Frames
                            │
                            ▼
                     Model Inference
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        EfficientTAM     nanoVLM       SmolVLM
             │              │              │
             ▼              ▼              ▼
       Segmentation      Vision-        Vision-
          Masks          Language       Language
                         Processing     Processing
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                     Output Frames

⸻

🎯 Main Focus

The primary segmentation-oriented model in this project is EfficientTAM.

nanoVLM and SmolVLM are included for lightweight multimodal experimentation and model comparison. They are not presented as the primary segmentation models.

⸻

🚀 Key Features

* 🎯 Frame-based video segmentation
* ⚡ Lightweight model experimentation
* 🧠 EfficientTAM for segmentation
* 👁️ nanoVLM for lightweight vision-language processing
* 🔬 SmolVLM for compact multimodal processing
* 🖼️ Individual frame inference
* 📊 IoU-based evaluation
* 🧩 Modular project architecture
* 📈 Model comparison
* 💾 Automatic output generation

⸻

🤖 Models

1. EfficientTAM

EfficientTAM is the primary segmentation model used in this project.

It is designed for efficient video object segmentation and tracking, with an emphasis on reducing computational and memory requirements.

In this project, video sequences are represented as ordered image frames. EfficientTAM processes these frames and generates the corresponding segmentation masks.

🔗 Official Repository

https://github.com/yformer/EfficientTAM

⸻

2. nanoVLM

nanoVLM is a lightweight Vision-Language Model developed by Hugging Face.

It is included in this project for lightweight multimodal experimentation and model comparison.

🔗 Resources

* GitHub: https://github.com/huggingface/nanoVLM
* Hugging Face Blog: https://huggingface.co/blog/nanovlm

Note: nanoVLM is a Vision-Language Model and is not treated as the primary segmentation model in this project.

⸻

3. SmolVLM

SmolVLM is a compact Vision-Language Model developed by Hugging Face.

It is included for lightweight multimodal processing and comparison with the other implemented models.

🔗 Resources

* Documentation: https://huggingface.co/docs/transformers/model_doc/smolvlm
* Hugging Face Blog: https://huggingface.co/blog/smolvlm

Note: SmolVLM is a Vision-Language Model and is not presented as the primary segmentation model.

⸻

📂 Dataset

The dataset used in this project consists of video sequences represented as pre-extracted image frames.

The original videos are converted into ordered frames before the segmentation stage.

Example Dataset Structure

dataset/
│
├── Video_01/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   └── ...
│
├── Video_02/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   └── ...
│
└── ...

This approach allows the segmentation pipeline to process the video data as an ordered sequence of frames while preserving the original temporal order.

Note: The original video files are not required during the segmentation stage because the dataset is already available as individual frames.

⸻

🔬 Processing Pipeline

The project follows a frame-based processing pipeline.

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
│    Model Inference       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Segmentation Mask      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     Processed Frame      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Output Result       │
└──────────────────────────┘

🎞️ Video Sequence Processing

Each video sequence is represented by an ordered set of frames:

Frame 001 ──► Model ──► Mask 001
Frame 002 ──► Model ──► Mask 002
Frame 003 ──► Model ──► Mask 003
    ⋮
Frame N   ──► Model ──► Mask N

The processed frames preserve the original frame order and can subsequently be used to visualize or reconstruct the segmented video.

⸻

📁 Project Structure

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

Project Components

File / Directory	Description
models/	Implementations of the supported models
models/efficienttam/	EfficientTAM implementation
models/nanovlm/	nanoVLM implementation
models/smolvlm/	SmolVLM implementation
utils/	Helper functions and utility modules
outputs/	Generated masks and processed results
sample_input/	Sample input frames
configs/	Model and project configuration files
main.py	Main project entry point
process_all_models.py	Script for running the implemented models
requirements.txt	Python project dependencies
README.md	Project documentation

⸻

⚙️ Installation

1. Clone the Repository

git clone https://github.com/bamimahyar/video-segmentation-lightweight.git
cd video-segmentation-lightweight

2. Install Dependencies

pip install -r requirements.txt

3. GPU Support

GPU execution is recommended for faster inference.

Make sure that the installed version of PyTorch is compatible with the CUDA version available on your system.

⸻

▶️ Usage

🖼️ Process a Single Frame

To process an individual input frame:

python process_all_models.py --image sample_input/00035.jpg

The implemented models process the input according to their respective implementations and generate the corresponding outputs.

⸻

🎞️ Process a Video Sequence

The project works with pre-extracted video frames rather than requiring an MP4 file during the segmentation stage.

The general workflow is:

                     Video
                       │
                       ▼
              Pre-extracted Frames
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Frame 1   Frame 2   Frame N
             │         │         │
             └─────────┼─────────┘
                       ▼
                 Model Processing
                       │
                       ▼
                Segmentation Masks
                       │
                       ▼
                 Processed Frames

For EfficientTAM, the ordered frames are processed as a video sequence and segmentation masks are generated for the corresponding frames.

⸻

📊 Evaluation

The primary segmentation metric considered in this project is Intersection over Union (IoU).

IoU measures the overlap between the predicted segmentation mask and the corresponding ground-truth mask.

IoU Formula

              Area of Prediction ∩ Ground Truth
IoU = ─────────────────────────────────────────────
              Area of Prediction ∪ Ground Truth

Evaluation Requirements

For a reproducible evaluation, the following are required:

* Predicted segmentation masks
* Corresponding ground-truth masks
* Consistent image dimensions
* A defined evaluation protocol
* The same dataset split for all models

## 📈 Evaluation Results

| Model | IoU ↑ | Inference Time (ms) ↓ | FPS ↑ |
|:------|------:|----------------------:|------:|
| **EfficientTAM** | TBD | TBD | TBD |
| **nanoVLM** | TBD | TBD | TBD |
| **SmolVLM** | TBD | TBD | TBD |

> **Metrics Legend:**
> - **IoU** — Intersection over Union (higher is better)
> - **Inference Time** — milliseconds per frame (lower is better)
> - **FPS** — Frames Per Second (higher is better)
📈 Model Comparison

The models are compared from several perspectives:

Criterion	EfficientTAM	nanoVLM	SmolVLM
Primary Purpose	Segmentation	Vision-Language	Vision-Language
Frame Processing	✓	✓	✓
Lightweight Focus	✓	✓	✓
Segmentation	✓	Experimental	Experimental
Multimodal Processing	—	✓	✓

The final quantitative comparison should be based on actual experiments performed using the same dataset, hardware configuration, and evaluation protocol.

⸻

⚠️ Limitations

* The dataset is provided as pre-extracted image frames.
* The segmentation stage does not require direct MP4 input.
* EfficientTAM is the primary segmentation-oriented model.
* nanoVLM and SmolVLM are primarily Vision-Language Models.
* Model performance depends on the dataset, model configuration, hardware, and preprocessing pipeline.
* Quantitative metrics should only be reported after evaluation against appropriate ground-truth masks.
* Results may vary depending on the hardware and inference configuration.

⸻

🔮 Future Work

Possible future improvements include:

* 🎯 More comprehensive quantitative evaluation
* 📊 Additional segmentation metrics such as Dice Score
* ⚡ FPS and inference-time benchmarking
* 💾 GPU memory usage comparison
* 🎞️ Automatic reconstruction of segmented videos
* 🧪 Evaluation on additional datasets
* 🔧 Further optimization of lightweight inference
* 📈 More extensive model comparison

⸻

📚 References

EfficientTAM

EfficientTAM — Efficient and memory-efficient video object segmentation and tracking.

Official Repository:
https://github.com/yformer/EfficientTAM

nanoVLM

Hugging Face — nanoVLM.

Official Repository:
https://github.com/huggingface/nanoVLM

Introduction:
https://huggingface.co/blog/nanovlm

SmolVLM

Hugging Face — SmolVLM.

Documentation:
https://huggingface.co/docs/transformers/model_doc/smolvlm

Introduction:
https://huggingface.co/blog/smolvlm

⸻

📌 Academic Use

This repository is intended for experimentation, research, and academic use involving lightweight video segmentation and multimodal vision models.

When using the implemented models in academic work, please cite the corresponding original papers and official repositories.

⸻

👨‍💻 Author

Mahyar Bami

GitHub:
https://github.com/bamimahyar

Project Repository:
https://github.com/bamimahyar/video-segmentation-lightweight

⸻

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Built for lightweight, practical, and research-oriented video segmenta