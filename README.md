# n-bits

Algorithms to better understand DNN (deep neural networks) weights.

This tool gathers bit usage in a ML model and prints statistics. It is quite performance optimized, processing
405 billions weights (over 800GiB) in slightly more than 3 minutes.


## Installation

```bash
go install github.com/maruel/n-bits-go/cmd/n-bits@latest
```


## Usage

Analyze popular models in increasingly large size:

```bash
# LLM
n-bits analyze -hf-repo Qwen/Qwen2.5-0.5B
n-bits analyze -hf-repo facebook/MobileLLM-1B
n-bits analyze -hf-repo meta-llama/Llama-3.2-1B
n-bits analyze -hf-repo HuggingFaceTB/SmolLM2-1.7B
n-bits analyze -hf-repo google/gemma-2-2b
n-bits analyze -hf-repo microsoft/Phi-3.5-mini-instruct
n-bits analyze -hf-repo mistralai/Mistral-7B-v0.3 -hf-glob model*.safetensors
n-bits analyze -hf-repo meta-llama/Llama-3.1-70B-Instruct
n-bits analyze -hf-repo meta-llama/Llama-3.1-405B-Instruct

# Image
n-bits analyze -hf-repo Qwen/Qwen2-VL-2B-Instruct
n-bits analyze -hf-repo stabilityai/stable-diffusion-3.5-large
n-bits analyze -hf-repo stabilityai/stable-diffusion-3.5-large -hf-glob text_encoder_3/*.safetensors
n-bits analyze -hf-repo stabilityai/stable-fast-3d
n-bits analyze -hf-repo meta-llama/Llama-3.2-11B-Vision
n-bits analyze -hf-repo meta-llama/Llama-Guard-3-11B-Vision
n-bits analyze -hf-repo black-forest-labs/FLUX.1-dev -hf-glob flux1-dev.safetensors

# Audio
n-bits analyze -hf-repo openai/whisper-large-v3-turbo
n-bits analyze -hf-repo openai/whisper-large-v3 -hf-glob model.safetensors
n-bits analyze -hf-repo openai/whisper-large-v3 -hf-glob model.fp32*.safetensors
```

The results range from 6.3% (openai/whisper-large-v3-turbo in float16), with SD3.5 being a close second at
6.4% to (openai/whisper-large-v3 in float32) 50% wasted. The median is around 17%.


### Metadata

Dump the metadata for each of the models you downloaded up to now:

```
find ~/.cache/huggingface/hub/models--*/snapshots -type l -name '*.safetensors' -exec n-bits metadata -name {} \;
```
