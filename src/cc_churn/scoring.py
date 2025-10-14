#!/usr/bin/env python3


"""Define helper functions for scoring model predictions."""

import sklearn.metrics as mtr


def false_positive_rate_scorer(y_true, y_pred):
    """
    Calculates the False Positive Rate (FPR).
    """
    tn, fp, fn, tp = mtr.confusion_matrix(y_true, y_pred).ravel()
    # Handle the case where FP + TN is zero to avoid division by zero
    if (fp + tn) == 0:
        return 0.0  # Or handle as appropriate for your use case
    return fp / (fp + tn)


def false_negative_rate_scorer(y_true, y_pred):
    """
    Calculates the False Negative Rate (FNR).

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    float
        The False Negative Rate.
    """
    tn, fp, fn, tp = mtr.confusion_matrix(y_true, y_pred).ravel()

    # Handle the case where (fn + tp) is zero to avoid division by zero
    if (fn + tp) == 0:
        return 0.0  # Or handle as appropriate for your use case
    else:
        return fn / (fn + tp)


def get_scorers(scorers_wanted):
    """."""
    scorers = {
        # use if labels are required and false negatives are more costly
        "f2": mtr.make_scorer(
            lambda y_true, y_pred: mtr.fbeta_score(y_true, y_pred, beta=2)
        ),
        # use if probabilities are required and positive class is more important
        "prauc": mtr.make_scorer(
            lambda y_true, y_pred: mtr.average_precision_score(y_true, y_pred)
        ),
        "recall": mtr.make_scorer(
            lambda y_true, y_pred: mtr.recall_score(y_true, y_pred)
        ),
        # a threshold-independent metric
        "rocauc": mtr.make_scorer(
            lambda y_true, y_pred: mtr.roc_auc_score(
                y_true, y_pred, average="macro"
            )
        ),
        # 'fpr': mtr.make_scorer(
        #     lambda y_true, y_pred: false_positive_rate_scorer(
        #         y_true, y_pred
        #     ), greater_is_better=False
        # ),
        # 'fnr': mtr.make_scorer(
        #     lambda y_true, y_pred: false_negative_rate_scorer(
        #         y_true, y_pred
        #     ), greater_is_better=False
        # ),
        "f05": mtr.make_scorer(
            lambda y_true, y_pred: mtr.fbeta_score(y_true, y_pred, beta=0.5)
        ),
        "f1": mtr.make_scorer(
            lambda y_true, y_pred: mtr.fbeta_score(y_true, y_pred, beta=1)
        ),
        "rocauc": mtr.make_scorer(
            lambda y_true, y_pred: mtr.roc_auc_score(y_true, y_pred)
        ),
        # include accuracy for informational purposes only
        "accuracy": mtr.make_scorer(
            lambda y_true, y_pred: mtr.accuracy_score(y_true, y_pred)
        ),
    }
    return {s: scorers[s] for s in scorers_wanted}
