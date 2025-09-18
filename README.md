# Efficient LoRA Fine-Tuning of BLIP-2 + LLaMA-2 for Visual Question Answering (VQA)

This repository provides code and examples for efficient parameter-efficient fine-tuning (PEFT) of BLIP-2 vision-language models with a LLaMA-2 7B backbone using LoRA adapters.

It supports:
- 4-bit quantized model loading with bitsandbytes for low memory footprint
- LoRA for fine-tuning on limited GPU (e.g. RTX 4050, Google Colab)
- VQA tasks using a subset of the VQA v2 dataset
- Interactive Colab notebook for image upload and question answering

## Setup

Install dependencies:

