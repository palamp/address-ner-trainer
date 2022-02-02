import json

from tensorflow.keras.models import load_model

from app.services.dataset import encode_character_input, padding_dataset


def test_load_model():
    with open("dataset_test.json", "r") as f:
        dataset = json.load(f)
    with open("ner_label_index.json", "r") as f:
        ner_label_index = json.load(f)
    token_c = encode_character_input(dataset["test_word"], 140, 10).astype("int32")
    dataset = padding_dataset(dataset, 140, ner_label_index)
    model = load_model("models/20220202-024852")
    model.summary()
    model.predict([dataset["test_word"], token_c])
    pass
