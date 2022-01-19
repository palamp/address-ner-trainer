from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import dill
import numpy as np
from numpy import full
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from . import thai2fit_model


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


def encode_character_input(sentences, max_len_word, max_len_char):
    char2idx = char_to_index()
    x_char = []
    for sentence in tqdm(sentences):
        sent_seq = full((max_len_word, max_len_char), char2idx["pad"])
        word_loop = len(sentence) if len(sentence) < max_len_word else max_len_word
        for i in range(word_loop):
            char_loop = len(sentence[i][0]) if len(sentence[i][0]) < max_len_char else max_len_char
            for j in range(char_loop):
                sent_seq[i][j] = char2idx.get(sentence[i][0][j], char2idx["unknown"])
        x_char.append(sent_seq)
    return x_char


def get_dataset(max_len) -> Dict[str, np.ndarray]:
    """
    Args:
        max_len(int): word padding
    Returns:
        A dictionary contains
            key: String
            value: Numpy array with shape `(len(sequences), maxlen)`
    """

    def sentences_to_word(sentences_dataset) -> Tuple[List, List]:
        word = []
        target = []
        for sentences in sentences_dataset:
            for word_tup in sentences:
                word.append(word_tup[0])
                target.append(word_tup[1])
        return word, target

    with open("dataset/ner.data", "rb") as file:
        datatofile = dill.load(file)
    train_sents, test_sents = train_test_split(datatofile, test_size=0.1, random_state=112)

    train_word, train_target = sentences_to_word(train_sents)
    test_word, test_target = sentences_to_word(test_sents)

    thai2dict_word_index = {word: index for index, word in enumerate(thai2fit_model.index2word)}
    ner_label_index = {label: index for index, label in enumerate(sorted(set(train_target + ["pad"])))}

    def prepare_sequence_word(input_text):
        idxs = list()
        for word in input_text:
            if word in thai2dict_word_index:
                idxs.append(thai2dict_word_index[word])
            else:
                idxs.append(thai2dict_word_index["unknown"])  # Use UNK tag for unknown word
        return idxs

    def prepare_sequence_target(input_label):
        idxs = [ner_label_index[w] for w in input_label]
        return idxs

    def prepare_sequence(data: List, func: Callable, pad_idx):
        result = pad_sequences(
            sequences=[func(s) for s in data],
            value=pad_idx,
            maxlen=max_len,
            padding="post",
            truncating="post",
        )
        return result

    train_word = prepare_sequence(train_word, prepare_sequence_word, thai2dict_word_index["pad"])
    train_target = prepare_sequence(train_target, prepare_sequence_target, ner_label_index["pad"])
    test_word = prepare_sequence(test_word, prepare_sequence_word, thai2dict_word_index["pad"])
    test_target = prepare_sequence(test_target, prepare_sequence_target, ner_label_index["pad"])

    return {
        "x_word": train_word,
        "x_target": train_target,
        "y_word": test_word,
        "y_target": test_target,
    }
