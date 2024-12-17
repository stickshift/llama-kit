import torch

from llama_kit.model import LlamaGenerator, load_config, load_parameters


def test_load_state_dict(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B", device=device)

    # I loaded parameters from checkpoint
    params = load_parameters(config)

    # I created a generator
    generator = LlamaGenerator(config)

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


def test_323b_text(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B", device=device)

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config))

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


def test_3211b_text(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 11B checkpoint
    config = load_config("Llama3.2-11B-Vision", device=device)

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config))

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


def test_323b_instruct(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B Instruct checkpoint
    config = load_config("Llama3.2-3B-Instruct", device=device)

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config))

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


def test_3211b_instruct(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 11B Instruct checkpoint
    config = load_config("Llama3.2-11B-Vision-Instruct", device=device)

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config))

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


def test_max_tokens(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B Instruct checkpoint
    config = load_config("Llama3.2-3B-Instruct", device=device)

    # I created a generator w/ max tokens of 10
    max_tokens = 10
    generator = LlamaGenerator(config, max_tokens=max_tokens)
    generator.load_state_dict(load_parameters(config))

    # I create an open ended prompt
    prompt = [
        {
            "role": "user",
            "content": "Tell me a story about dragons.",
        }
    ]

    #
    # Whens
    #

    # I generate response
    tokens = []
    for token in generator(prompt):
        assert len(tokens) < max_tokens
        tokens.append(token)

    #
    # Thens
    #

    # token should have max_tokens elements
    assert len(tokens) == max_tokens
