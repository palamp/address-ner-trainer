import re
from typing import Dict

emoji_pattern = re.compile(r"[\u0E00-\u0E7Fa-zA-Z0-9' \n\-!$%^&*()_+|~=`{}\[\]:'<>?,.#\/]")


def replace_and_remove_tags(text):
    text = (
        text.strip()
        .replace("FACILITY", "LOCATION")
        .replace("[AGO]", "")
        .replace("[/AGO]", "")
        .replace("[T]", "")
        .replace("[/T]", "")
    )
    text = re.sub("<[^>]*>", "", text)
    return text


def normalize_quote(text):
    return text.replace("''", '"').replace("’", '"').replace("‘", '"')


def normalize_space(text):
    text = text.replace("\t", "")
    text = text.replace("\u200b", "")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    return text.strip()


def remove_emojis(text):
    return "".join(emoji_pattern.findall(text))


def remove_tag(text, cache: Dict = {}):
    """remove tag such as; `[LOCATION]`, `[PERSON]`"""
    if text in cache:
        return cache[text]
    cache[text] = re.sub("\[(.*?)\]", "", text)
    return cache[text]


def is_unique(text_line, unique: set):
    text = re.sub("<[^>]*>", "", text_line)
    text = remove_tag(text)
    # text = re.sub("\[\/(.*?)\]", "", text)
    if text not in unique:
        unique.add(text)
        return True
    else:
        return False
