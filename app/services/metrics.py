from itertools import chain
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow_addons.metrics import FBetaScore
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked


def convert_model_prediction(y_true, y_pred, ner_label_index):
    index_ner_label = {v: k for k, v in ner_label_index.items()}

    y_true_cvt = []
    y_pred_cvt = []
    for ind in range(0, len(y_pred)):
        try:
            out = y_pred[ind]
            true = y_true[ind]
            revert_pred = [index_ner_label[i] for i in out]
            revert_true = [index_ner_label[i] for i in true]
            y_pred_cvt.append(revert_pred)
            y_true_cvt.append(revert_true)
        except:
            print(ind)
    return (np.array(y_true_cvt), np.array(y_pred_cvt))


def ner_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    tagset = list(sorted(set(lb.classes_)))
    tagset = [i for i in tagset if len(i.split("-")) == 2]
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    print(list(sorted(set(lb.classes_))))
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        digits=4,
    )


class F1Score(FBetaScore):
    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        average: str = None,
        threshold: Optional[FloatTensorLike] = None,
        name: str = "f1_score",
        dtype: AcceptableDTypes = None,
    ):
        self.num_classes = num_classes
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true = tf.one_hot(y_true, self.num_classes)
        y_pred = tf.one_hot(y_pred, self.num_classes)
        for idx in tf.range(tf.shape(y_true)[0]):
            super(F1Score, self).update_state(y_true[idx], y_pred[idx], None)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
