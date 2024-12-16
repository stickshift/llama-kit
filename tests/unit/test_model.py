import torch

from llama_kit.model import LlamaModel, load_config, load_parameters, load_tokenizer, unpack_parameters


def test_transform(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # I created a model
    model = LlamaModel(config, device)

    # I loaded model parameters
    params = load_parameters(config, map_location=device)
    model.load_state_dict(unpack_parameters(params, ["model"]))

    # Greek prompt
    prompt = "alpha beta gamma"

    # I split prompt into tokens
    tokenizer = load_tokenizer(config)
    token_ids = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), device=device)

    #
    # Whens
    #

    # I transform token ids into semantic embeddings
    embeddings = model(token_ids)

    #
    # Thens
    #

    # embeddings should be n x d_model
    assert embeddings.shape[0] == len(token_ids)
    assert embeddings.shape[1] == config.d_model
