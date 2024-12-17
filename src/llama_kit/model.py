import json
from pathlib import Path
from typing import Iterator, Mapping, NamedTuple, Sequence, override

from llama_models.llama3.api import Tokenizer
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tools import default_arg, torch_device
from enum import Enum

__all__ = [
    "LlamaAttention",
    "LlamaCausalLMHead",
    "LlamaFFN",
    "LlamaGenerator",
    "LlamaHead",
    "LlamaLayer",
    "LlamaModel",
    "Message",
    "ModelConfig",
    "ModelParameters",
    "Tokenizer",
    "load_config",
    "load_parameters",
    "load_tokenizer",
    "masked_attention_bias",
    "render_prompt",
    "rope_frequencies",
    "rope_rotate",
    "rope_swap",
    "unpack_parameters",
]


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


class ModelType(str, Enum):
    PRETRAINED = "pretrained"
    INSTRUCT = "instruct"


class ModelConfig(NamedTuple):
    """Llama3 model config."""

    checkpoint_path: Path

    type: ModelType

    vocab_size: int

    d_model: int

    d_head: int

    d_ffn: int

    n_layers: int

    n_heads: int

    n_kv_heads: int

    rms_norm_eps: float

    rope_theta: float

    device: torch.device

    dtype: torch.dtype


def load_config(checkpoint_name: str, **kwargs) -> ModelConfig:
    """Load Llama3 config from checkpoint params.json."""
    # Build checkpoint_path
    checkpoints_path = Path("~/.llama/checkpoints").expanduser()
    checkpoint_path = checkpoints_path / checkpoint_name

    # Load hyperparameters
    hparams_path = checkpoint_path / "params.json"
    hparams = json.loads(hparams_path.read_text())

    # Calculate d_ffn from 8/3 * d_model rounded to nearest multiple_of
    d_model = hparams["dim"]
    ffn_dim_multiplier = hparams["ffn_dim_multiplier"]
    multiple_of = hparams["multiple_of"]
    d_ffn = int(8 / 3 * d_model * ffn_dim_multiplier)
    d_ffn = multiple_of * ((d_ffn + multiple_of - 1) // multiple_of)

    data = {
        "checkpoint_path": checkpoint_path,
        "type": ModelType.INSTRUCT if checkpoint_name.endswith("-Instruct") else ModelType.PRETRAINED,
        "vocab_size": hparams["vocab_size"],
        "d_model": hparams["dim"],
        "n_layers": hparams["n_layers"],
        "rms_norm_eps": hparams["norm_eps"],
        "n_heads": hparams["n_heads"],
        "d_head": int(hparams["dim"] / hparams["n_heads"]),
        "n_kv_heads": hparams["n_kv_heads"],
        "rope_theta": hparams["rope_theta"],
        "d_ffn": d_ffn,
        "device": torch_device(),
        "dtype": torch.bfloat16,
    }

    # Override with kwargs
    data |= kwargs

    return ModelConfig(**data)


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

ModelParameters = Mapping[str, Tensor]
"""Maps parameter names to weights."""


def load_parameters(config: ModelConfig, **kwargs) -> ModelParameters:
    """Load model state from checkpoint."""
    # Load state from checkpoint
    checkpoint_params = torch.load(
        config.checkpoint_path / "consolidated.00.pth",
        weights_only=True,
        **kwargs,
    )

    # Remap Meta's parameter names
    params = {}

    # Inject prefix for multimodal checkpoints
    prefix = "text_model." if "text_model.tok_embeddings.weight" in checkpoint_params else ""

    # Validate dtype
    assert checkpoint_params[f"{prefix}tok_embeddings.weight"].dtype == config.dtype

    # Embeddings
    params |= {
        "model.embeddings.weight": checkpoint_params[f"{prefix}tok_embeddings.weight"],
    }

    # Layers
    for layer_id in range(config.n_layers):
        params |= {
            f"model.layers.{layer_id}.attention.normalize.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.attention_norm.weight"
            ],
            f"model.layers.{layer_id}.attention.w_queries.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.attention.wq.weight"
            ],
            f"model.layers.{layer_id}.attention.w_keys.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.attention.wk.weight"
            ],
            f"model.layers.{layer_id}.attention.w_values.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.attention.wv.weight"
            ],
            f"model.layers.{layer_id}.attention.w_output.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.attention.wo.weight"
            ],
            f"model.layers.{layer_id}.ffn.normalize.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.ffn_norm.weight"
            ],
            f"model.layers.{layer_id}.ffn.w_input.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.feed_forward.w3.weight"
            ],
            f"model.layers.{layer_id}.ffn.w_gate.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.feed_forward.w1.weight"
            ],
            f"model.layers.{layer_id}.ffn.w_output.weight": checkpoint_params[
                f"{prefix}layers.{layer_id}.feed_forward.w2.weight"
            ],
        }

    # Head
    params |= {
        "head.normalize.weight": checkpoint_params[f"{prefix}norm.weight"],
        "head.w_output.weight": checkpoint_params[f"{prefix}output.weight"],
    }

    return params


def unpack_parameters(params: ModelParameters, path: Sequence[str]) -> ModelParameters:
    """Extract component parameters."""
    # Calculate component prefix, e.g. model.layers.0.norm
    prefix = ".".join(path) + "."

    # Extract relevant keys and strip prefix
    component_params = {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}

    return component_params


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------


def load_tokenizer(config: ModelConfig) -> Tokenizer:
    """Load tokenizer from checkpoint."""
    # Load tiktoken model
    return Tokenizer(str(config.checkpoint_path / "tokenizer.model"))


# ------------------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------------------


class LlamaEmbeddings(nn.Embedding):
    """Llama token embeddings layer."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=config.device,
            dtype=config.dtype,
        )


# ------------------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------------------


def rope_frequencies(config: ModelConfig, n: int):
    """Compute RoPE cos and sin rotation matrices."""
    # Hyperparameters
    base = config.rope_theta
    d = config.d_head
    device = config.device
    dtype = config.dtype

    # Calculate thetas
    i = torch.arange(d // 2, dtype=dtype).to(device)
    thetas = base ** (-2 * i / d)

    # Duplicate each theta, e.g. [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
    thetas = thetas.repeat_interleave(2)

    # Repeat thetas for each position from 0 to n and stack in an (n, d_head) matrix
    theta_stack = torch.stack([m * thetas for m in range(n)])

    # Apply cos, sin
    r_cos = torch.cos(theta_stack)
    r_sin = torch.sin(theta_stack)

    # Sanity check
    assert r_cos.shape[0] == n and r_cos.shape[1] == config.d_head  # noqa: PT018
    assert r_sin.shape[0] == n and r_sin.shape[1] == config.d_head  # noqa: PT018

    return r_cos, r_sin


def rope_swap(x):
    """Maps [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""
    # Preserve original shape
    s = x.shape

    # Split into pairs, swap, and restore shape
    x = x.reshape(-1, 2).flip(-1).view(s)

    # Multiply every even index along the last dimension by -1
    #   e.g. [x0, x1, x2, x3] -> [-x0, x1, -x2, x3]
    x[..., ::2] *= -1

    return x


def rope_rotate(x, r_cos, r_sin):
    """Rotate embeddings using RoPE transform."""
    return (x * r_cos) + (rope_swap(x) * r_sin)


def masked_attention_bias(config: ModelConfig, n: int) -> Tensor:
    """Compute reusable masked attention bias term M.

    Returns:
        Tensor: (n, n) diagonal matrix w/ upper triangular elements set to -inf, 0 otherwise.
    """
    from torch import logical_not, ones, tril, zeros  # noqa: PLC0415

    # Parameters
    device = config.device
    dtype = config.dtype

    # Initialize m with zeros
    m = zeros(n, n, device=device, dtype=dtype)

    # Create boolean mask w/ main diagonal and below set to False
    mask = logical_not(tril(ones(n, n, device=device, dtype=torch.bool)))

    # Fill upper triangular region to -inf
    m = m.masked_fill_(mask, float("-inf"))

    return m


class LlamaNorm(torch.nn.Module):
    """Normalizes embeddings using RMSNorm.

    See RMSNorm in llama-models.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(
            torch.ones(
                config.d_model,
                device=config.device,
                dtype=config.dtype,
            )
        )

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight


class LlamaAttention(nn.Module):
    """Llama attention layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Input normalization
        self.normalize = LlamaNorm(config)

        # Queries projection
        self.w_queries = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_heads * config.d_head,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

        # Keys projection
        self.w_keys = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

        # Values projection
        self.w_values = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

        # Output projection
        self.w_output = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor, mask: Tensor) -> Tensor:
        # Save residuals
        residual = x

        # Normalize inputs
        x = self.normalize(x)

        # Project inputs to query, key, value spaces
        q = self.w_queries(x)
        k = self.w_keys(x)
        v = self.w_values(x)

        # Split attention heads
        q = self._split_heads(q, self.config.n_heads)
        k = self._split_heads(k, self.config.n_kv_heads)
        v = self._split_heads(v, self.config.n_kv_heads)

        # Expand key/value groups
        reps = self.config.n_heads // self.config.n_kv_heads
        k = k.repeat_interleave(reps, dim=0)
        v = v.repeat_interleave(reps, dim=0)

        # Encode positions by rotating queries and keys
        q = rope_rotate(q, r_cos, r_sin)
        k = rope_rotate(k, r_cos, r_sin)

        # Compute attention for all heads in parallel
        scores = q @ k.transpose(-2, -1) / np.sqrt(self.config.d_head) + mask
        a = F.softmax(scores, dim=-1) @ v

        # Combine attention heads
        a = self._combine_heads(a)

        # Project outputs back to model space
        a = self.w_output(a)

        # Merge outputs with residuals
        x = residual + a

        return x

    def _split_heads(self, x: Tensor, n_heads: int):
        """Split attention heads."""
        return x.view(-1, n_heads, self.config.d_head).transpose(-3, -2)

    def _combine_heads(self, x):
        """Combine attention heads."""
        return x.transpose(-3, -2).contiguous().view(-1, int(self.config.n_heads * self.config.d_head))


# ------------------------------------------------------------------------------
# FFN
# ------------------------------------------------------------------------------


class LlamaFFN(nn.Module):
    """Llama feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Input normalization
        self.normalize = LlamaNorm(config)

        # Input projection
        self.w_input = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

        # Gate projection
        self.w_gate = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

        # Output projection
        self.w_output = nn.Linear(
            in_features=config.d_ffn,
            out_features=config.d_model,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Save residuals
        residual = x

        # Normalize inputs
        x = self.normalize(x)

        # Apply SwiGLU transform
        f = F.silu(self.w_gate(x)) * self.w_input(x)

        # Project outputs back to model space
        f = self.w_output(f)

        # Merge outputs with residuals
        x = residual + f

        return x


# ------------------------------------------------------------------------------
# Layer
# ------------------------------------------------------------------------------


class LlamaLayer(nn.Module):
    """Llama transformer layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention = LlamaAttention(config)

        self.ffn = LlamaFFN(config)

    @override
    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor, mask: Tensor) -> Tensor:
        # Attention
        x = self.attention(x, r_cos, r_sin, mask)

        # FFN
        x = self.ffn(x)

        return x


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------


class LlamaModel(nn.Module):
    """Combines embeddings and layers in reusable module."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embeddings = LlamaEmbeddings(config)

        self.layers = nn.ModuleList(LlamaLayer(config) for _ in range(config.n_layers))

    @override
    def forward(self, token_ids: Tensor) -> Tensor:
        # Precompute values shared across layers:
        #
        #   * RoPE rotation matrices
        #   * Masked attention bias
        #

        n = len(token_ids)

        # RoPE rotation matrices
        r_cos, r_sin = rope_frequencies(self.config, n)

        # Masked attention bias
        mask = masked_attention_bias(self.config, n)

        # Map tokens to embeddings
        x = self.embeddings(token_ids)

        # Transform token embeddings to semantic embeddings
        for layer in self.layers:
            x = layer(x, r_cos, r_sin, mask)

        return x


# ------------------------------------------------------------------------------
# Head
# ------------------------------------------------------------------------------


class LlamaHead(nn.Module):
    """Llama prediction head."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Input normalization
        self.normalize = LlamaNorm(config)

        # Output projection
        self.w_output = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Normalize inputs
        x = self.normalize(x)

        # Use last embedding to represent the entire sequence
        x = x[-1]

        # Project outputs to token space
        x = self.w_output(x)

        return x


class LlamaCausalLMHead(LlamaHead):
    """Llama causal language model head."""

    def __init__(
        self,
        config: ModelConfig,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        super().__init__(config)

        self.temperature = default_arg(temperature, 0.6)
        self.top_k = default_arg(top_k, 50)
        self.top_p = default_arg(top_p, 0.9)

    @override
    def forward(self, x: Tensor) -> int:
        # Project semantic embeddings to token space
        x = super().forward(x)

        # Temperature
        # -----------

        # If temperature is 0, return the top token
        if self.temperature == 0:
            return torch.argmax(x, dim=-1).item()

        # Apply temperature
        x = x / self.temperature

        # Ranking
        # -------

        # Convert logits to probabilities
        probs = F.softmax(x, dim=-1)

        # Sort probabilities in descending order
        probs, indices = probs.sort(descending=True)

        # Top K
        # -----

        # Retain top k tokens
        probs = probs[: self.top_k]

        # Top P
        # -----

        # Find cutoff where cumulative probability exceeds top_p
        cumulative_mask = probs.cumsum(dim=-1) > self.top_p
        threshold_index = torch.argmax(cumulative_mask).item()

        # Only apply threshold if top_p was exceeded
        if cumulative_mask.any():
            probs = probs[: threshold_index + 1]

        # Random Selection
        # ----------------

        # Sample from remaining tokens weighted by probability
        sampled_index = torch.multinomial(probs, 1)

        # Convert sampled_index to original logits
        token_id = indices[sampled_index]

        return token_id.item()


# ------------------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------------------


class Message(NamedTuple):
    """Structured message for instruct model."""

    role: str

    content: str


def render_prompt(config: ModelConfig, messages: Sequence[Message]) -> str:
    """Render messages."""
    prompt = ""

    for data in messages:
        message = data

        # Convert dicts to Messages
        if isinstance(message, dict):
            message = Message(**message)

        if config.type == ModelType.INSTRUCT:
            prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"

        prompt += message.content
    
        if config.type == ModelType.PRETRAINED:
            prompt += "\n\n"
    
        if config.type == ModelType.INSTRUCT:
            prompt += "<|eot_id|>"

    if config.type == ModelType.INSTRUCT:
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


class LlamaGenerator(nn.Module):
    """Llama text generator."""

    def __init__(
        self,
        config: ModelConfig,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ):
        super().__init__()

        self.config = config

        self.tokenizer = load_tokenizer(config)

        self.max_tokens = default_arg(max_tokens, 32)

        self.model = LlamaModel(config)

        self.head = LlamaCausalLMHead(
            config,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def __call__(self, prompt: str | Sequence[Message], **kwargs) -> Iterator[str]:
        """Generate token ids until stop token or we exceed max tokens."""
        # Prepare model
        self.model.eval()
        self.head.eval()

        # Encode messages
        if not isinstance(prompt, str):
            prompt = render_prompt(self.config, prompt)

        # Tokenize
        token_ids = self.tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")

        # Override fields with kwargs
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        with torch.no_grad():
            # Generate output until we get a stop token or we exceed max_tokens.
            for _ in range(max_tokens):
                # Load token ids into a tensor
                x = torch.tensor(token_ids, device=self.config.device, dtype=torch.int64)

                # Transform token_ids into semantic embeddings
                x = self.model(x)

                # Predict next token
                token_id = self.head(x)

                # Check stopping criteria
                if token_id in self.tokenizer.stop_tokens:
                    break

                # Yield token
                yield self.tokenizer.decode([token_id])

                # Append to end of sequence
                token_ids.append(token_id)
