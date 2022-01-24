import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
from tensorflow_addons.text.crf import crf_log_likelihood

thai2fit_model: KeyedVectors = KeyedVectors.load_word2vec_format("thai2fit/thai2vecNoSym.bin", binary=True)


class ConditionalRandomFieldLoss(object):
    def __init__(self, name: str = "crf_loss"):
        self.name = name

    def get_config(self):
        return {"name": self.name}

    def __call__(self, y_true, y_pred, sample_weight=None):
        crf_layer = y_pred._keras_history[0]

        # check if last layer is CRF
        if not isinstance(crf_layer, CRF):
            raise ValueError("Last layer must be CRF for use {}.".format(self.__class__.__name__))
        _, potentials, sequence_length, chain_kernel = y_pred
        loss_vector = self.compute_crf_loss(potentials, sequence_length, chain_kernel, y_pred, sample_weight)

        return loss_vector

    def compute_crf_loss(self, potentials, sequence_length, kernel, y, sample_weight=None):
        crf_likelihood, _ = crf_log_likelihood(potentials, y, sequence_length, kernel)
        # convert likelihood to loss
        flat_crf_loss = -1 * crf_likelihood
        if sample_weight is not None:
            flat_crf_loss = flat_crf_loss * sample_weight
        crf_loss = tf.reduce_mean(flat_crf_loss)

        return crf_loss


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
    max_len_word=284,
    max_len_char=30,
    main_lstm_unit=256,  # Bidirectional 256 + 256 = 512
    lstm_recurrent_dropout=0.5,
):
    # Word Input and Word Embedding Using Thai2Fit
    word_in = Input(shape=(max_len_word,), name="word_input")
    word_embeddings = Embedding(
        input_dim=n_thai2dict,
        output_dim=400,
        weights=[thai2fit_model.vectors],
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

    # CRF
    out = CRF(crf_unit)(main_lstm)

    # Model
    model = Model(inputs=[word_in, char_in], outputs=out)

    model.compile(optimizer="adam", loss=ConditionalRandomFieldLoss(), metrics=[Accuracy()])

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
    is_early_stop=False,
):
    filepath = "saved_model/weights-improvement-{epoch:02d}-{accuracy:.3f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    early_stopper = EarlyStopping(patience=5, restore_best_weights=True)
    callbacks_list = [checkpoint]
    if is_early_stop:
        callbacks_list.append(early_stopper)

    history = model.fit(
        [X_word_tr, np.array(X_char_tr).reshape(-1, max_len, max_len_char)],
        y_tr,
        batch_size=train_batch_size,
        epochs=1,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(
            [X_word_te, np.array(X_char_te).reshape(-1, max_len, max_len_char)],
            y_te,
        ),
        shuffle=True,
    )

    return history, model
