#!/bin/bash
set -eu

# Warning: This downloads more than 1TiB of data.

# LLM
time n-bits analyze -hf-repo Qwen/Qwen2.5-0.5B
time n-bits analyze -hf-repo facebook/MobileLLM-1B
time n-bits analyze -hf-repo meta-llama/Llama-3.2-1B
time n-bits analyze -hf-repo HuggingFaceTB/SmolLM2-1.7B
time n-bits analyze -hf-repo google/gemma-2-2b
time n-bits analyze -hf-repo microsoft/Phi-3.5-mini-instruct
time n-bits analyze -hf-repo mistralai/Mistral-7B-v0.3 -hf-glob model*.safetensors
time n-bits analyze -hf-repo meta-llama/Llama-3.1-70B-Instruct
time n-bits analyze -hf-repo meta-llama/Llama-3.1-405B-Instruct

# Image
time n-bits analyze -hf-repo Qwen/Qwen2-VL-2B-Instruct
time n-bits analyze -hf-repo stabilityai/stable-diffusion-3.5-large
time n-bits analyze -hf-repo stabilityai/stable-diffusion-3.5-large -hf-glob text_encoder_3/*.safetensors
time n-bits analyze -hf-repo stabilityai/stable-fast-3d
time n-bits analyze -hf-repo meta-llama/Llama-3.2-11B-Vision
time n-bits analyze -hf-repo meta-llama/Llama-Guard-3-11B-Vision
time n-bits analyze -hf-repo black-forest-labs/FLUX.1-dev -hf-glob flux1-dev.safetensors

# Audio
time n-bits analyze -hf-repo openai/whisper-large-v3-turbo
time n-bits analyze -hf-repo openai/whisper-large-v3 -hf-glob model.safetensors
time n-bits analyze -hf-repo openai/whisper-large-v3 -hf-glob model.fp32*.safetensors
