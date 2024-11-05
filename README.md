# n-bits

Algorithms to better understand DNN (deep neural networks) weights.


## Installation

```bash
go install github.com/maruel/n-bits-go/cmd/n-bits@latest
```


## Usage

Analyze both Llama-3.2 1B and Qwen2.5 0.5B:

```bash
n-bits analyze -hf-repo meta-llama/Llama-3.2-1B
n-bits analyze -hf-repo Qwen/Qwen2.5-0.5B
```
