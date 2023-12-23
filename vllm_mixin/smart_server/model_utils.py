import sys
import os


def get_stop(args):
    if "qwen" in args.model.lower():
        return "<|endoftext|>"
    elif "baichuan" in args.model.lower():
        return "</s>"
    else:
        return None


def get_dialogue_indicators(args):
    if "qwen" in args.model.lower():
        user_indicator = "<|im_start|>### Instruction:\n"
        ai_indicator = "### Response:\n"
    elif "baichuan" in args.model.lower():
        user_indicator = "### Instruction:\n"
        ai_indicator = "### Response:\n"
    else:
        user_indicator = ""
        ai_indicator = ""
    return user_indicator, ai_indicator


def get_prompt(
        prompt,
        histories,
        user_indicator="### Instruction:\n",
        ai_indicator="### Response:\n",
        sep="\n\n",
        partial=False
):
    convs = ""
    sub_convs = ""

    # drop the previous conversations if user has summarized
    for history in histories:
        history_prompt = history[0]
        history_response = history[1]
        sub_convs = sub_convs + f"""{user_indicator}{history_prompt}\n\n{ai_indicator}{history_response}{sep}"""

    sub_convs = sub_convs + f"""{user_indicator}{prompt}\n\n{ai_indicator}"""
    convs = convs + sub_convs
    if partial:
        return sub_convs, len(sub_convs)
    else:
        return convs, len(sub_convs)
