import dill
from pytest import fixture
from sklearn.model_selection import train_test_split

from services.models.dataset import char_to_index, encode_character_input


def test_char_to_index():
    output = char_to_index()
    assert all([isinstance(v, int) for v in output.values()])
    assert all([isinstance(k, str) for k in output])


@fixture(scope="session")
def load_file():
    with open("dataset/ner.data", "rb") as file:
        datatofile = dill.load(file)

    train_sentence, test_sentence = train_test_split(datatofile, test_size=0.1, random_state=112)
    return (train_sentence, test_sentence)


@fixture(scope="session")
def character_input(load_file):
    return {"sentences": load_file[0], "max_len": 284, "max_len_char": 30}


def test_encode_character_input(character_input):
    x = encode_character_input(**character_input)
    pass
