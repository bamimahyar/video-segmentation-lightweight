# 🎯 Video Segmentation with Lightweight Models

> Master's Thesis Project

## 📊 Results

| Model | IoU | Dice | Accuracy |
|-------|-----|------|----------|
| EfficientTAM 🏆 | 0.2276 | 0.3388 | 0.7654 |
| nanoVLM | 0.0144 | 0.0183 | 0.8144 |
| SmolVLM | 0.0072 | 0.0091 | 0.8125 |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python process_all_models.py --image sample_input/00035.jpg
🤖 Models

    EfficientTAM - Best model (IoU: 0.2276)

    nanoVLM - Lightweight model

    SmolVLM - Transformer model

📁 Structure
models/          # Model code
utils/           # Helper functions
outputs/         # Generated results
sample_input/    # Test images
📧 Contact

Developer: Mahyar Bami
GitHub: bamimahyar

⭐ Give a star if useful!
