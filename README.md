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

Models :

    EfficientTAM - Best model (IoU: 0.2276)

    nanoVLM - Lightweight model

    SmolVLM - Transformer model

Project Structure : 
models/          # Code for segmentation models
utils/           # Helper functions and utilities
outputs/         # Generated results and outputs
sample_input/    # Sample test images
configs/         # Configuration files
main.py          # Main execution script
process_all_models.py # Script to run all models
requirements.txt # Project dependencies

References and Citations :

This project builds upon the following foundational works. Please cite them if you use this code:

EfficientTAM : 

    Paper: "Efficient Video Transformers with Temporal Attention" (Example placeholder)

    Official Repository: [Link to the official EfficientTAM repo if exists]

    Citation :
        @article{efficienttam2024,
          author = {Author1, A. and Author2, B.},
          title = {Efficient Video Transformers with Temporal Attention},
          journal = {Conference on Computer Vision},
          year = {2024}
        }

nanoVLM :

    Paper: "Nano-scale Vision-Language Models for Edge Devices" (Example placeholder)

    Official Repository: [Link to official nanoVLM repo]

    Citation:
        @inproceedings{nanovlm2023,
          author = {Author3, C. and Author4, D.},
          title = {Nano-scale Vision-Language Models for Edge Devices},
          booktitle = {NeurIPS Workshop on Efficient AI},
          year = {2023}
        }

SmolVLM

    Paper: "SmolVLM: Tiny yet Mighty Vision-Language Models" (Example placeholder)

    Official Repository: [Link to official SmolVLM repo]

    Citation:
        @misc{smolvlm2023,
          author = {Author5, E. and Author6, F.},
          title = {SmolVLM: Tiny yet Mighty Vision-Language Models},
          year = {2023},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/example/smolvlm}}
        }

Contact

    Developer: Mahyar Bami

    GitHub: bamimahyar
