import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
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
from tensorflow.keras.metrics import Accuracy
from tensorflow_addons.layers.crf import CRF

from crf.crf import CRF
from crf.crf_losses import crf_loss

from . import thai2fit_model


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


def create_models(
    n_thai2dict,
    n_chars,
    crf_unit,
    max_len=284,
    max_len_char=30,
    main_lstm_unit=256,  # Bidirectional 256 + 256 = 512
    lstm_recurrent_dropout=0.5,
):
    # Word Input and Word Embedding Using Thai2Fit
    word_in = Input(shape=(max_len,), name="word_input")
    word_embeddings = Embedding(
        input_dim=n_thai2dict,
        output_dim=400,
        weights=[thai2fit_model.vectors],
        input_length=max_len,
        mask_zero=False,
        name="word_embedding",
        trainable=False,
    )(word_in)

    # Character Input and Character Embedding
    char_in = Input(shape=(max_len, max_len_char), name="char_input")
    char_embeddings = CharacterEmbeddingBlock(n_chars, max_len_char, name="char_embedding")(char_in)

    # Concatenate All Embedding
    all_word_embeddings = SpatialDropout1D(0.3)(concatenate([word_embeddings, char_embeddings]))

    # Main Model Dense attention
    main_lstm = Bidirectional(
        LSTM(units=main_lstm_unit, return_sequences=True, recurrent_dropout=lstm_recurrent_dropout)
    )(all_word_embeddings)
    main_lstm = TimeDistributed(Dense(50, activation="relu"))(main_lstm)

    # CRF
    out = CRF(crf_unit)(main_lstm)

    # Model
    model = Model(inputs=[word_in, char_in], outputs=out)

    model.compile(optimizer="adam", loss=crf_loss, metrics=[Accuracy()])

    model.summary()
    return model


def fit_model(
    model: Model,
    X_word_tr,
    X_char_tr,
    y_tr,
    X_word_te,
    X_char_te,
    y_te,
    train_batch_size=32,
    max_len=284,
    max_len_char=30,
):
    filepath = "saved_model/weights-improvement-{epoch:02d}-{accuracy:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    callbacks_list = [checkpoint]

    history = model.fit(
        [X_word_tr, np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
        y_tr,
        batch_size=train_batch_size,
        epochs=1,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(
            [X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))],
            y_te,
        ),
        shuffle=True,
    )

    return history, model
