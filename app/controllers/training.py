from datetime import datetime
from itertools import chain
from pathlib import Path

from ..services.dataset import char_to_index, encode_character_input, get_dataset, padding_dataset
from ..services.metrics import convert_model_prediction, ner_classification_report
from ..services.model import create_models, fit_model, thai2fit_model


def train_model_controller(debug=False, early_stop=False):
    n_word = len(thai2fit_model.index2word)
    n_char = len(char_to_index())
    # NOTE: for calculaation (mode, median, mean, sd) = (12, 36, 49.37, 42.50)
    max_len_word = 140
    # NOTE: for calculaation (mode, median, mean, sd) = (3, 3, 4.08, 2.91)
    max_len_char = 10

    dataset = get_dataset("dataset/ner.data")
    train_char = encode_character_input(dataset["train_word"], max_len_word, max_len_char)
    test_char = encode_character_input(dataset["test_word"], max_len_word, max_len_char)

    ner_label_index = {
        ner: idx
        for idx, ner in enumerate(
            ["padding"]
            + sorted(
                set(chain.from_iterable(dataset["train_target"])), key=lambda x: x[2:] if len(x) > 1 else x[0]
            )
        )
    }
    dataset = padding_dataset(dataset, max_len_word, ner_label_index)

    model = create_models(
        n_word,
        n_char,
        len(ner_label_index),
        max_len_word=max_len_word,
        max_len_char=max_len_char,
        debug=debug,
    )
    save_dir = Path("saved_model/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    save_dir.mkdir(parents=True, exist_ok=True)
    history, model = fit_model(
        model,
        dataset["train_word"],
        train_char,
        dataset["train_target"],
        dataset["test_word"],
        test_char,
        dataset["test_target"],
        save_dir,
        is_early_stop=early_stop,
    )

    pred_model = model.predict([dataset["test_word"], test_char], verbose=1)
    report = ner_classification_report(
        *convert_model_prediction(dataset["test_target"], pred_model, ner_label_index)
    )
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)
