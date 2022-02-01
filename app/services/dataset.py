from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import dill
import numpy as np
from numpy import full
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from .model import thai2fit_model


@lru_cache
def char_to_index() -> Dict[str, int]:
    # fmt: off
    special_char = [
        "padding", "unknown", " ", "$", "#", "!", "%", "&", "*", "+", ",", "-", ".", "/", ":", ";", "?",
        "@", "^", "_", "`", "=", "|", "~", "'",'"', "(", ")", "{", "}", "<", ">", "[", "]", "\n"
    ]
    # fmt: on

    len_special_char = len(special_char)
    special_char_dict = {char: idx for idx, char in enumerate(special_char)}

    chars = {w_i for word in thai2fit_model.index2word for w_i in word}
    char_idx = {char: idx + len_special_char for idx, char in enumerate(chars)}

    result = {**special_char_dict, **char_idx}
    return result


@lru_cache
def word_to_index() -> Dict[str, int]:
    word_index = {word: index + 1 for index, word in enumerate(thai2fit_model.index2word)}
    # reserved 0 value to `padding`
    word_index["padding"] = 0
    return word_index


def encode_character_input(sentences, max_len_word, max_len_char):
    char2idx = char_to_index()
    x_char = []
    for sentence in tqdm(sentences):
        sent_seq = full((max_len_word, max_len_char), char2idx["padding"])
        word_loop = len(sentence) if len(sentence) < max_len_word else max_len_word
        for i in range(word_loop):
            char_loop = len(sentence[i]) if len(sentence[i]) < max_len_char else max_len_char
            for j in range(char_loop):
                sent_seq[i][j] = char2idx.get(sentence[i][j], char2idx["unknown"])
        x_char.append(sent_seq)
    return np.array(x_char).reshape(-1, max_len_word, max_len_char)


def get_dataset(filepath) -> Dict[str, np.ndarray]:
    """
    Returns:
        A dictionary length = 4, [train_word, train_target, test_word, test_target]
    """

    def sentences_to_word(sentences_dataset) -> Tuple[List, List]:
        word_sent = []
        target_sent = []
        for sentences in sentences_dataset:
            word = []
            target = []
            for word_tup in sentences:
                word.append(word_tup[0])
                target.append(word_tup[1])
            word_sent.append(word)
            target_sent.append(target)
        return word_sent, target_sent

    with open(filepath, "rb") as file:
        datatofile = dill.load(file)
    train_sents, test_sents = train_test_split(datatofile, test_size=0.1, random_state=112)

    train_word, train_target = sentences_to_word(train_sents)
    test_word, test_target = sentences_to_word(test_sents)

    return {
        "train_word": train_word,
        "train_target": train_target,
        "test_word": test_word,
        "test_target": test_target,
    }


def padding_dataset(
    dataset: Dict[str, np.ndarray], max_len: int, ner_label_index: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """
    Args:
        dataset(dict):
            A dictionary length = 4, contains: train_word, train_target, test_word, test_target
        max_len(int): word padding
    Returns:
        A dictionary length = 4, [train_word, train_target, test_word, test_target]
            key: String
            value: Numpy array with shape `(len(sequences), maxlen)`
    """

    word_index = word_to_index()

    def encode_word(input_text):
        idxs = [word_index.get(word, word_index["unknown"]) for word in input_text]
        return idxs

    def encode_target(input_label):
        idxs = [ner_label_index[w] for w in input_label]
        return idxs

    def padding_sequence(data: List, func: Callable, pad_idx):
        result = pad_sequences(
            sequences=[func(s) for s in data],
            value=pad_idx,
            maxlen=max_len,
            padding="post",
            truncating="post",
        )
        return result

    train_word = padding_sequence(dataset["train_word"], encode_word, word_index["padding"])
    train_target = padding_sequence(dataset["train_target"], encode_target, ner_label_index["padding"])
    test_word = padding_sequence(dataset["test_word"], encode_word, word_index["padding"])
    test_target = padding_sequence(dataset["test_target"], encode_target, ner_label_index["padding"])

    return {
        "train_word": train_word,
        "train_target": train_target,
        "test_word": test_word,
        "test_target": test_target,
    }
