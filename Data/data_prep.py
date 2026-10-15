def make_prompt(sentence: str, prompt_template: str) -> str:
    """
    Replaces {{insert}} with the input sentence, to make a complete prompt.

    Args:
        sentence (str): The input sentence.
        prompt_template (str): a Template with {{insert}} to get a complete prompt

    Returns:
        str: the combined str.
    """
    return prompt_template.replace("{{insert}}" , sentence)