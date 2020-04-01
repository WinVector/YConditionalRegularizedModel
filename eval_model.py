import numpy

import pandas

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# noinspection PyPep8Naming
def eval_model(*, model, X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    preds_out = numpy.zeros(X.shape[0])
    fold_id = numpy.zeros(X.shape[0])
    accuracy_results = numpy.zeros(n_splits)
    i = 0
    for train_index, test_index in kfold.split(X, y):
        model.fit(X[train_index, :], y[train_index])
        pi = model.predict_proba(X[test_index, :])
        accuracy = accuracy_score(y_true=y[test_index] > 0.5, y_pred=pi[:, 1] > 0.5)
        accuracy_results[i] = accuracy
        preds_out[test_index] = pi[:, 1]
        fold_id[test_index] = i
        i = i + 1
    res = pandas.DataFrame({"prediction": preds_out, "true_y": y, "fold_id": fold_id})
    return res, accuracy_results
