from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras import Model, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Embedding,
    Input,
    Layer,
    SpatialDropout1D,
    TimeDistributed,
    concatenate,
)
from tensorflow_addons.text.crf_wrapper import CRFModelWrapper

from .metrics import F1Score

thai2fit_model: KeyedVectors = KeyedVectors.load_word2vec_format("app/statics/thai2vecNoSym.bin", binary=True)


class CharacterEmbeddingBlock(Layer):
    def __init__(
        self,
        n_chars,
        max_len_char,
        character_embedding_dim=32,
        character_LSTM_unit=32,
        lstm_recurrent_dropout=0.5,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._config = {
            "n_chars": n_chars,
            "max_len_char": max_len_char,
            "character_embedding_dim": character_embedding_dim,
            "character_LSTM_unit": character_LSTM_unit,
            "lstm_recurrent_dropout": lstm_recurrent_dropout,
            "name": name,
        }
        self.embedding = TimeDistributed(
            Embedding(
                input_dim=n_chars,
                output_dim=character_embedding_dim,
                input_length=max_len_char,
                mask_zero=False,
            )
        )

        # Character Sequence to Vector via BiLSTM
        self.encoding = TimeDistributed(
            Bidirectional(
                LSTM(
                    units=character_LSTM_unit,
                    return_sequences=False,
                    recurrent_dropout=lstm_recurrent_dropout,
                )
            )
        )

    def call(self, inputs):
        x = self.embedding(inputs)
        return self.encoding(x)

    def get_config(self):
        return self._config


def create_models(
    n_thai2dict,
    n_chars,
    crf_unit,
    max_len_word,
    max_len_char,
    main_lstm_unit=256,  # Bidirectional 256 + 256 = 512
    lstm_recurrent_dropout=0.5,
    debug=False,
):
    # add weight for padding
    weights = np.insert(thai2fit_model.vectors, 0, np.zeros(400), axis=0)

    # Word Input and Word Embedding Using Thai2Fit
    word_in = Input(shape=(max_len_word,), name="word_input")
    word_embeddings = Embedding(
        input_dim=n_thai2dict + 1,
        output_dim=400,
        weights=[weights],
        input_length=max_len_word,
        mask_zero=False,
        name="word_embedding",
        trainable=False,
    )(word_in)

    # Character Input and Character Embedding
    char_in = Input(shape=(max_len_word, max_len_char), name="char_input")
    char_embeddings = CharacterEmbeddingBlock(n_chars, max_len_char, name="char_embedding")(char_in)

    # Concatenate All Embedding
    all_word_embeddings = SpatialDropout1D(0.3)(concatenate([word_embeddings, char_embeddings]))

    # Main Model Dense attention
    main_lstm = Bidirectional(
        LSTM(units=main_lstm_unit, return_sequences=True, recurrent_dropout=lstm_recurrent_dropout)
    )(all_word_embeddings)
    main_lstm = TimeDistributed(Dense(50, activation="relu"))(main_lstm)

    # Model
    base_model = Model(inputs=[word_in, char_in], outputs=main_lstm)

    model = CRFModelWrapper(base_model, crf_unit)
    model.compile(
        optimizer="adam",
        metrics=[
            metrics.Accuracy(),
            # F1Score(crf_unit, "weighted"),
        ],
        run_eagerly=debug,
    )
    return model


def fit_model(
    model: Model,
    X_word_tr,
    X_char_tr,
    y_tr,
    X_word_te,
    X_char_te,
    y_te,
    save_dir: Path,
    train_batch_size=32,
    epoch=100,
    is_early_stop=False,
):
    checkpoint_path = save_dir / "weights-improvement-{epoch:02d}-{val_loss:.3f}.ckpt"
    tensorboard_logpath = save_dir / "tensorboard"
    model_savepath = save_dir / "last_weight"

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    early_stopper = EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=5, restore_best_weights=True, verbose=1, mode="min"
    )
    tensorboard = TensorBoard(
        log_dir=tensorboard_logpath,
        histogram_freq=1,
        write_steps_per_second=True,
        update_freq=8,
        embeddings_freq=1,
    )
    callbacks_list = [checkpoint, tensorboard]
    if is_early_stop:
        callbacks_list.append(early_stopper)

    history = model.fit(
        [X_word_tr, X_char_tr],
        y_tr,
        batch_size=train_batch_size,
        epochs=epoch,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=([X_word_te, X_char_te], y_te),
        shuffle=True,
    )
    model.save(model_savepath)

    return history, model
