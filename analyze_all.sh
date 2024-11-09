#!/bin/bash
# Warning: This downloads more than 1TiB of data.

set -eu
cd "$(dirname $0)"

go install ./cmd/n-bits
mkdir -p data

analyze() {
  REPO=$1
  shift
  echo "- Analyzing $REPO"
  time n-bits analyze -hf-repo $REPO -json "data/$(basename $REPO).json" "$@"
}

# LLM
analyze Qwen/Qwen2.5-0.5B
analyze facebook/MobileLLM-1B
analyze meta-llama/Llama-3.2-1B
analyze HuggingFaceTB/SmolLM2-1.7B
analyze google/gemma-2-2b
analyze microsoft/Phi-3.5-mini-instruct
analyze mistralai/Mistral-7B-v0.3 -hf-glob model*.safetensors
analyze meta-llama/Llama-3.1-70B-Instruct
analyze meta-llama/Llama-3.1-405B-Instruct

# Image
analyze Qwen/Qwen2-VL-2B-Instruct
analyze stabilityai/stable-diffusion-3.5-large
analyze stabilityai/stable-diffusion-3.5-large -hf-glob text_encoder_3/*.safetensors
analyze stabilityai/stable-fast-3d
analyze meta-llama/Llama-3.2-11B-Vision
analyze meta-llama/Llama-Guard-3-11B-Vision
analyze black-forest-labs/FLUX.1-dev -hf-glob flux1-dev.safetensors

# Audio
analyze openai/whisper-large-v3-turbo
analyze openai/whisper-large-v3 -hf-glob model.safetensors
analyze openai/whisper-large-v3 -hf-glob model.fp32*.safetensors
