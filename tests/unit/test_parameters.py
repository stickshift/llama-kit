import torch

from llama_kit.model import load_config, load_parameters


def test_load_parameters(device: torch.device):
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    #
    # Whens
    #

    # I load parameters from checkpoint
    params = load_parameters(config)

    #
    # Thens
    #

    # params should include LlamaTransformer keys
    assert "model.embeddings.weight" in params

    for layer_id in range(config.n_layers):
        assert f"model.layers.{layer_id}.attention.normalize.weight" in params
        assert f"model.layers.{layer_id}.attention.w_queries.weight" in params
        assert f"model.layers.{layer_id}.attention.w_keys.weight" in params
        assert f"model.layers.{layer_id}.attention.w_values.weight" in params
        assert f"model.layers.{layer_id}.attention.w_output.weight" in params
        assert f"model.layers.{layer_id}.ffn.normalize.weight" in params
        assert f"model.layers.{layer_id}.ffn.w_input.weight" in params
        assert f"model.layers.{layer_id}.ffn.w_gate.weight" in params
        assert f"model.layers.{layer_id}.ffn.w_output.weight" in params

    assert "head.normalize.weight" in params
    assert "head.w_output.weight" in params

    # tensors should be loaded to device
    assert params["model.embeddings.weight"].device.type == device.type
