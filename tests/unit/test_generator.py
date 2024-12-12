import torch

from llama_kit.model import LlamaGenerator, load_config, load_parameters, load_tokenizer


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


def test_generate(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # I created a tokenizer
    tokenizer = load_tokenizer(config)

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, device, stop_tokens=tokenizer.stop_tokens, temperature=0)

    # I loaded state from checkpoint
    generator.load_state_dict(load_parameters(config, map_location=device))

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I split prompt into tokens
    token_ids = tokenizer.encode(prompt, bos=True, eos=False)

    # I generate next token
    token_id = next(generator(token_ids))

    #
    # Thens
    #

    # token should be "delta"
    assert tokenizer.decode([token_id]).strip() == "delta"
