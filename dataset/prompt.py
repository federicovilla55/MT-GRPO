def prompt_mod(prompt_to_change):
    prompt_to_ret = """
    Please rewrite the following sentence so that it becomes 
    slightly more complex and harder to understand, while still 
    remaining logical and roughly the same length. Do not change the meaning. 
    Only increase the linguistic complexity. Sentence you have to change: """+prompt_to_change+""" 
    """
    return prompt_to_ret