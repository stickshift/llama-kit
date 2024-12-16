import torch

from llama_kit.model import LlamaGenerator, load_config, load_parameters


def test_load_state_dict(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # I loaded parameters from checkpoint
    params = load_parameters(config, map_location=device)

    # I created a generator
    generator = LlamaGenerator(config, device=device)

    #
    # Whens
    #

    # I load state from checkpoint
    generator.load_state_dict(params)

    #
    # Thens
    #

    # state should match checkpoint
    assert torch.equal(generator.model.embeddings.get_parameter("weight"), params["model.embeddings.weight"])


def test_generate_text_model(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, device, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config, map_location=device))

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I generate next token
    token = next(generator(prompt))

    #
    # Thens
    #

    # token should be "delta"
    assert token.strip() == "delta"


def test_generate_instruct_model(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B Instruct checkpoint
    config = load_config("Llama3.2-3B-Instruct")

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, device, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config, map_location=device))

    # Boston prompt
    prompt = [
        {
            "role": "user",
            "content": "What is the capital of Massachusetts? Answer in one word.",
        }
    ]

    #
    # Whens
    #

    # I generate next token
    token = next(generator(prompt))

    #
    # Thens
    #

    # token should be "Boston"
    assert token.strip() == "Boston"
