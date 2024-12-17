from pathlib import Path
from random import sample

from llama_kit.benchmarks.mmlu import download_dataset, load_dataset, select_question, generate_prompt, evaluate_model


def test_323b_pretrained(mmlu_dataset_path: Path):
    """Benchmarks Llama 3.2 3B pretrained on sample of MMLU dataset.
    
    See https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md for comparison.
    """

    #
    # Givens
    #

    # Sample size of 16
    n = 16

    # I loaded dataset
    questions, examples, _ = load_dataset(mmlu_dataset_path)

    # I sampled questions
    selected = sample(questions, k=n)

    #
    # Whens
    #

    # I evaluate 323b model using 5-shot prompts
    score = evaluate_model("Llama3.2-3B", questions=selected, n_shots=5, examples=examples)
    
    #
    # Thens
    #

    # score should be ~ 0.58 according to model card
    assert score > 0.5
