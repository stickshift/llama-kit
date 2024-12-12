# Llama Development Kit

> Lightweight collection of reusable Llama building blocks optimized for experimentation.

## Components

The `llama_kit.model` package provides the following PyTorch modules.

* `LlamaGenerator`
* `LlamaHead`
* `LlamaCausalLMHead`
* `LlamaModel`
* `LlamaLayer`
* `LlamaAttention`
* `LlamaFFN`
* `LlamaEmbeddings`

## Prerequisites

* Python 3.12
* uv

## Build and Test

```shell
# Configure environment
source environment.sh

# Build and test everything
make tests
```
