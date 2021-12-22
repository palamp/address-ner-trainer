from functools import lru_cache
from typing import Dict

from numpy import full
from tqdm import tqdm

from . import thai2fit_model

# def word_to_index():
#     thai2dict = {}
#     for word in thai2fit_model.index2word:
#         thai2dict[word] = thai2fit_model[word]
#     return thai2dict


@lru_cache
def char_to_index() -> Dict[str, int]:
    # fmt: off
    special_char = [
        "pad", "unknown", " ", "$", "#", "!", "%", "&", "*", "+", ",", "-", ".", "/", ":", ";", "?",
        "@", "^", "_", "`", "=", "|", "~", "'",'"', "(", ")", "{", "}", "<", ">", "[", "]", "\n"
    ]
    # fmt: on

    len_special_char = len(special_char)
    special_char_dict = {char: idx for idx, char in enumerate(special_char)}

    chars = {w_i for word in thai2fit_model.index2word for w_i in word}
    char_idx = {char: idx + len_special_char for idx, char in enumerate(chars)}

    result = {**special_char_dict, **char_idx}
    return result


def encode_character_input(sentences, max_len, max_len_char):
    char2idx = char_to_index()
    x_char = []
    for sentence in tqdm(sentences):
        sent_seq = full((max_len, max_len_char), char2idx["pad"])
        word_loop = len(sentence) if len(sentence) < max_len else max_len
        for i in range(word_loop):
            char_loop = len(sentence[i][0]) if len(sentence[i][0]) < max_len_char else max_len_char
            for j in range(char_loop):
                sent_seq[i][j] = char2idx.get(sentence[i][0][j], char2idx["unknown"])
        x_char.append(sent_seq)
    return x_char
