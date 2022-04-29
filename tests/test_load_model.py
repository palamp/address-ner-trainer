import pickle
from pathlib import Path

import pytest
from tensorflow.keras.models import load_model


@pytest.fixture(scope="session")
def last_model_dir():
    return max(Path("saved_model").iterdir(), key=lambda x: x.stat().st_mtime if x.is_dir() else 0)


def test_load_model(last_model_dir: Path):
    def assertion_pickle(pkl, extra_index=[]):
        if not isinstance(extra_index, list):
            extra_index = [extra_index, "padding"]
        else:
            extra_index.append("padding")
        assert isinstance(pkl, dict)
        assert all([pkl.get(index) is not None for index in extra_index])
        assert list(sorted(pkl.values())) == list(range(len(pkl)))

    assert (last_model_dir / "last_weight").exists()
    assert (last_model_dir / "statics/ner_label_index.pickle").exists()
    assert (last_model_dir / "statics/char_index.pickle").exists()
    assert (last_model_dir / "statics/word_index.pickle").exists()
    assert (last_model_dir / "statics/max_len_word_char.pickle").exists()

    with open(last_model_dir / "statics/ner_label_index.pickle", "rb") as nerdict:
        ner_label_index = pickle.load(nerdict)
    assertion_pickle(ner_label_index)

    with open(last_model_dir / "statics/char_index.pickle", "rb") as chardict:
        char_index = pickle.load(chardict)
    assertion_pickle(char_index, "unknown")

    with open(last_model_dir / "statics/word_index.pickle", "rb") as f:
        word_index = pickle.load(f)
    assertion_pickle(word_index, "unknown")

    with open(last_model_dir / "statics/max_len_word_char.pickle", "rb") as f:
        max_len_word, max_len_char = pickle.load(f)
    assert isinstance(max_len_word, int)
    assert isinstance(max_len_char, int)

    model = load_model(last_model_dir / "last_weight")
    assert model.built
