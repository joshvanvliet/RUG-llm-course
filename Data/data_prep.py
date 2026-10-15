from pathlib import Path
from typing import Literal

from yaml import safe_load


def make_prompt(sentence: str, prompt_name: Literal["instruction", "conversation"]) -> str:
    """
    Replaces {{insert}} with the input sentence, to make a complete prompt.

    Args:
        sentence (str): The input sentence.
        prompt_template (str): a Template with {{insert}} to get a complete prompt

    Returns:
        str: the combined str.
    """
    with open(Path("./routing.yml")) as routing_file:
        routes = safe_load(routing_file)

        with open(Path(routes["prompt_template"])) as prompt_template_file:
            prompt_template: str = safe_load(prompt_template_file)[prompt_name]
    
    return prompt_template.replace("{{insert}}", sentence)