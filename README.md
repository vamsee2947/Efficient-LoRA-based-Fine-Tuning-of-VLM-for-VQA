# Efficient LoRA Fine-Tuning of BLIP-2 + LLaMA-2 for Visual Question Answering (VQA)

This repository provides code and examples for efficient parameter-efficient fine-tuning (PEFT) of BLIP-2 vision-language models with a LLaMA-2 7B backbone using LoRA adapters.

It supports:
- 4-bit quantized model loading with bitsandbytes for low memory footprint
- LoRA for fine-tuning on limited GPU (e.g. RTX 4050, Google Colab)
- VQA tasks using a subset of the VQA v2 dataset
- Interactive Colab notebook for image upload and question answering

## Setup

Install dependencies:
pip install -r requirements.txt


## Usage

- Launch training with your dataset using `scripts/train.py` and config files
- Run inference and demos in `notebooks/vqa_demo.ipynb`
- Upload your images and ask questions interactively in the notebook

## Repository structure

- `scripts/train.py`: Main training script with LoRA integration
- `configs/vqa_lora.yaml`: Training and model configuration
- `notebooks/vqa_demo.ipynb`: Colab demo notebook for VQA inference
- `utils/preprocess.py`: Utilities for data preprocessing
- `requirements.txt`: Python dependencies list

---

