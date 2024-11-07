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
n-bits analyze -hf-repo black-forest-labs/FLUX.1-dev
n-bits analyze -hf-repo meta-llama/Llama-3.1-405B-Instruct
```

The average is between 12% and 20% wasted.
