from itertools import chain

from ..services.dataset import char_to_index, encode_character_input, get_dataset, padding_dataset
from ..services.model import create_models, fit_model, thai2fit_model


def train_model_controller(is_padding=False):
    n_word = len(thai2fit_model.index2word)
    n_char = len(char_to_index())
    max_len_word = 284
    max_len_char = 30

    dataset = get_dataset("dataset/ner.data")
    x_char_dataset = encode_character_input(dataset["train_word"], max_len_word, max_len_char)
    y_char_dataset = encode_character_input(dataset["test_word"], max_len_word, max_len_char)

    ner_label_index = {
        ner: idx
        for idx, ner in enumerate(sorted(set(chain.from_iterable(dataset["train_target"] + [["pad"]]))))
    }
    if is_padding:
        dataset = padding_dataset(dataset, max_len_word, ner_label_index)

    model = create_models(n_word, n_char, len(ner_label_index), max_len_word=max_len_word)
    history, model = fit_model(
        model,
        dataset["train_word"],
        x_char_dataset,
        dataset["train_target"],
        dataset["test_word"],
        y_char_dataset,
        dataset["test_target"],
    )

    save_filepath = "saved_model/last_weight-50.hdf5"
    model.save_weights(save_filepath)
