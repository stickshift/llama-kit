from textwrap import dedent

import torch

from llama_kit.model import load_config, render_prompt


def test_render_prompt_pretrained():
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # messages has system and user messages
    messages = [
        {"role": "system", "content": "alpha"},
        {"role": "user", "content": "beta"},
    ]

    #
    # Whens
    #

    # I render prompt
    prompt = render_prompt(config, messages)

    #
    # Thens
    #

    # prompt should be 
    assert prompt == "alpha\n\nbeta\n\n"


def test_render_prompt_instruct():
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B Instruct checkpoint
    config = load_config("Llama3.2-3B-Instruct")

    # messages has system and user messages
    messages = [
        {"role": "system", "content": "alpha"},
        {"role": "user", "content": "beta"},
    ]

    #
    # Whens
    #

    # I render prompt
    prompt = render_prompt(config, messages)

    #
    # Thens
    #

    # prompt should be 
    assert prompt == dedent(
        """
        <|start_header_id|>system<|end_header_id|>
        
        alpha<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        beta<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        """
    ).lstrip()
    