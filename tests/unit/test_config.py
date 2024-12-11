from llama_kit.model import load_config


def test_load_config():

    #
    # Givens
    #

    #
    # Whens
    #

    # I load config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    #
    # Thens
    #

    # d_model should be 3072
    assert config.d_model == 3072
