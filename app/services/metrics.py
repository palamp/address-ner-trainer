from itertools import chain

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


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
