import torch

from llama_kit.model import LlamaGenerator, generate_text, load_config, load_parameters, load_tokenizer


def test_generate_text(device: torch.device):
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

    # I append 1 new token to prompt
    for token in generate_text(tokenizer, generator, prompt, max_tokens=1):
        prompt += token

    #
    # Thens
    #

    # prompt should be "alpha beta gamma delta"
    assert prompt == "alpha beta gamma delta"
