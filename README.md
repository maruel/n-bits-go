# n-bits

Algorithms to better understand DNN (deep neural networks) weights.


## Installation

```bash
go install github.com/maruel/n-bits-go/cmd/n-bits@latest
```


## Usage

Analyze popular models in increasingly large size:

```bash
n-bits analyze -hf-repo Qwen/Qwen2.5-0.5B
n-bits analyze -hf-repo meta-llama/Llama-3.2-1B
n-bits analyze -hf-repo openai/whisper-large-v3-turbo
n-bits analyze -hf-repo openai/whisper-large-v3 -hf-glob model.safetensors
n-bits analyze -hf-repo openai/whisper-large-v3 -hf-glob model.fp32*.safetensors
n-bits analyze -hf-repo black-forest-labs/FLUX.1-dev -hf-glob flux1-dev.safetensors
n-bits analyze -hf-repo meta-llama/Llama-3.1-405B-Instruct
```

The results range from 6.3% (openai/whisper-large-v3-turbo in float16) to (openai/whisper-large-v3 in
float32) 50% wasted, with a median around 17%.
