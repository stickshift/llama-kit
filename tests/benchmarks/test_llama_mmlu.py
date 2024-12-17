from pathlib import Path
from random import sample

from llama_kit.model import load_config, load_parameters, load_tokenizer, render_prompt, unpack_parameters, LlamaHead, LlamaModel, ModelConfig
from llama_kit.benchmarks.mmlu import load_dataset, evaluate_generator, MMLUGenerator


def test_323b_instruct(mmlu_dataset_path: Path):
    """Benchmarks Llama 3.2 3B Instruct on sample of MMLU dataset.
    
    See https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md for comparison.
    """

    #
    # Givens
    #

    # Sample size
    n = 32

    # I loaded dataset
    dataset = load_dataset(mmlu_dataset_path)

    # I sampled questions
    questions = sample(dataset.questions, k=n)

    # I initialized generator
    config = load_config("Llama3.2-3B-Instruct")
    generator = MMLUGenerator(config)
    generator.load_state_dict(load_parameters(config))

    #
    # Whens
    #

    # I evaluate generator using 0-shot prompts
    score = evaluate_generator(generator, questions=questions, n_shots=0, examples=dataset.examples)
    
    #
    # Thens
    #

    # score should be ~ 0.58 according to model card
    assert score > 0.5
