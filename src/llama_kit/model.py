from functools import partial
import json
from pathlib import Path
from typing import Iterator, Mapping, NamedTuple, Protocol, Sequence, override

from llama_models.llama3.api import Tokenizer as LlamaTokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tools import default_arg

__all__ = [
    "ModelConfig",
    "load_config",
]


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


class ModelConfig(NamedTuple):
    """Llama3 model config."""

    checkpoint_path: Path

    vocab_size: int

    d_model: int

    d_head: int

    d_ffn: int

    n_layers: int

    n_heads: int

    n_kv_heads: int

    rms_norm_eps: float

    rope_theta: float


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
        "vocab_size": hparams["vocab_size"],
        "d_model": hparams["dim"],
        "n_layers": hparams["n_layers"],
        "rms_norm_eps": hparams["norm_eps"],
        "n_heads": hparams["n_heads"],
        "d_head": int(hparams["dim"] / hparams["n_heads"]),
        "n_kv_heads": hparams["n_kv_heads"],
        "rope_theta": hparams["rope_theta"],
        "d_ffn": d_ffn,
    }

    # Override with kwargs
    data |= kwargs

    return ModelConfig(**data)
