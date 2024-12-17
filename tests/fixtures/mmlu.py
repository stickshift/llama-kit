from pathlib import Path

import pytest

from llama_kit.benchmarks.mmlu import download_dataset

__all__ = [
    "mmlu_dataset_path",
]


@pytest.fixture
def mmlu_dataset_path(datasets_path: Path) -> Path:
    path = datasets_path / "mmlu"

    download_dataset(path)
    
    return path
