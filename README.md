# Image Segmentation with EfficientTAM, nanoVLM and SmolVLM

This project implements and compares multiple modern vision models for image segmentation, including **EfficientTAM, nanoVLM, and SmolVLM**.

## 🚀 Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

To process an input image using all available models:

```bash
python process_all_models.py --image sample_input/00035.jpg
```

> **Note:** If you are using Windows PowerShell or CMD, use the command above exactly as written. Avoid using `\_` in Python filenames or paths.

---

## 🤖 Models

### EfficientTAM

The primary and best-performing model used in this project.

* Model: **EfficientTAM**
* Task: Video/Image Object Segmentation
* Best observed IoU: **0.2276**

### nanoVLM

A lightweight Vision-Language Model designed for efficient vision-language tasks.

### SmolVLM

A compact Vision-Language Model designed to provide strong vision-language capabilities with a relatively small model size.

---

## 📁 Project Structure

```text
project/
│
├── models/                    # Segmentation and vision model implementations
│
├── utils/                     # Helper functions and utility modules
│
├── outputs/                   # Generated segmentation results and outputs
│
├── sample_input/              # Sample input images
│
├── configs/                   # Configuration files
│
├── main.py                    # Main execution script
│
├── process_all_models.py      # Script for running all available models
│
├── requirements.txt           # Project dependencies
│
└── README.md                  # Project documentation
```

---

## 📊 Model Evaluation

The models can be evaluated using segmentation metrics such as **Intersection over Union (IoU)**.

The current best observed result in this project is:

```text
EfficientTAM
IoU: 0.2276
```

> The reported IoU value depends on the dataset, input images, model configuration, preprocessing, and evaluation procedure.

---

## 📚 References and Citations

This project builds upon publicly available research and open-source implementations.

### EfficientTAM

EfficientTAM is a memory-efficient adaptation of the Segment Anything Model 2 (SAM 2) for efficient video object segmentation.

**Paper:**

> *EfficientTAM: Fast and Memory-Efficient Tracking and Segmentation*

**Official Repository:**

[EfficientTAM GitHub Repository](https://github.com/yformer/EfficientTAM?utm_source=chatgpt.com)

**Citation:**

```bibtex
@article{yang2024efficienttam,
  title={EfficientTAM: Fast and Memory-Efficient Tracking and Segmentation},
  author={Yang, Fan and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

### nanoVLM

nanoVLM refers to lightweight Vision-Language Model implementations designed for experimentation and efficient multimodal inference.

> Please use the citation provided by the specific nanoVLM implementation/repository used in this project, since the name "nanoVLM" is used by multiple projects.

---

### SmolVLM

SmolVLM is a compact family of Vision-Language Models developed by Hugging Face.

**Official Repository:**

[SmolVLM on Hugging Face](https://huggingface.co/blog/smolvlm?utm_source=chatgpt.com)

**Citation:**

```bibtex
@misc{shukla2025smolvlm,
  title={SmolVLM: Redefining small and efficient multimodal models},
  author={Shukla, Nikhil and others},
  year={2025},
  publisher={Hugging Face}
}
```

---

## 🛠️ Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

For GPU acceleration, make sure that your installed PyTorch version is compatible with your CUDA version.

---

## 💻 Usage

### Process a single image

```bash
python process_all_models.py --image sample_input/00035.jpg
```

### Using another image

```bash
python process_all_models.py --image sample_input/your_image.jpg
```

Generated results will be stored in:

```text
outputs/
```

---

## 📈 Results

The project is designed to compare the performance and characteristics of the implemented models.

| Model        | Type                  | Purpose                           |
| ------------ | --------------------- | --------------------------------- |
| EfficientTAM | Segmentation Model    | Main segmentation model           |
| nanoVLM      | Vision-Language Model | Lightweight multimodal processing |
| SmolVLM      | Vision-Language Model | Compact multimodal processing     |

Current best observed segmentation result:

```text
EfficientTAM
IoU = 0.2276
```

---

## ⚠️ Notes

* Model performance may vary depending on the input image and dataset.
* GPU execution is recommended for faster inference.
* Make sure all required model weights and configuration files are available before running the project.
* The reported IoU should be interpreted according to the evaluation dataset and ground-truth masks used by the project.

---

## 📧 Contact

**Developer:** Mahyar Bami

**GitHub:** [bamimahyar](https://github.com/bamimahyar?utm_source=chatgpt.com)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

Thank you for checking out the project!
