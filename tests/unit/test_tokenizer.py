from llama_kit.model import load_config, load_tokenizer


def test_tokenizer():
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    model_config = load_config("Llama3.2-3B")

    # I created a tokenizer
    tokenizer = load_tokenizer(model_config)

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I encode prompt
    token_ids = tokenizer.encode(prompt, bos=True, eos=False)

    #
    # Thens
    #

    # token_ids should be [128000, 7288, 13746, 22350]
    assert token_ids == [128000, 7288, 13746, 22350]

    #
    # Whens
    #

    # I decode last token id
    token = tokenizer.decode([token_ids[-1]])

    #
    # Thens
    #

    # token should be gamma
    assert token.strip() == "gamma"
