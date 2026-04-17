#!/usr/bin/env python3


"""Define helper functions for scoring model predictions."""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import sklearn.metrics as mtr


def custom_roc_auc(y_true: pd.Series, y_pred_proba: pd.Series) -> float:
    """Calculates the Area Under the Receiver Operating Characteristic Curve.

    Wraps scikit-learn's roc_auc_score to compute the AUC-ROC from probability
    scores.

    Args:
        y_true: Array-like of shape (n_samples,) with true binary labels.
        y_pred_proba: Array-like of shape (n_samples,) with predicted
            probabilities of positive class.

    Returns:
        float: The Area under the ROC curve.

    Raises:
        ValueError: If `y_true` and `y_pred` have incompatible shapes.

    Examples:
        >>> false_negative_rate_scorer([0, 1, 1], [0, 0, 1])
        0.5
    """
    return mtr.roc_auc_score(y_true, y_pred_proba)


def get_scorers(
    scorers_wanted: List[str],
) -> Dict[str, Callable]:
    """Builds a dictionary of sklearn-compatible scoring functions.

    This function returns a subset of predefined scoring metrics based
    on the requested metric names. Each scorer is constructed using
    `sklearn.metrics.make_scorer` and is compatible with model
    evaluation utilities such as cross-validation.

    Available scorers include:
        - "f2": F-beta score with beta=2 (recall-focused).
        - "prauc": Precision-recall AUC (average precision).
        - "recall": Recall score.
        - "f05": F-beta score with beta=0.5 (precision-focused).
        - "f1": F1 score (balanced precision/recall).
        - "rocauc": ROC AUC score.
        - "accuracy": Classification accuracy.

    Args:
        scorers_wanted: List of metric names to include in the output.

    Returns:
        Dict[str, Callable]: Dictionary mapping metric names to
            sklearn scorer callables.

    Raises:
        KeyError: If any requested scorer is not defined.

    Examples:
        >>> scorers = get_scorers(["f2", "recall", "accuracy"])
        >>> list(scorers.keys())
        ['f2', 'recall', 'accuracy']
    """
    scorers = {
        # use if labels are required and false negatives are more costly
        "f2": mtr.make_scorer(mtr.fbeta_score, beta=2),
        # use if probabilities are needed and +ve class is more important
        "prauc": mtr.make_scorer(mtr.average_precision_score),
        "recall": mtr.make_scorer(mtr.recall_score, zero_division=np.nan),
        "f05": mtr.make_scorer(mtr.fbeta_score, beta=0.5),
        "f1": mtr.make_scorer(mtr.fbeta_score, beta=1),
        "rocauc": mtr.make_scorer(
            custom_roc_auc, response_method="predict_proba"
        ),
        # use accuracy for informational purposes only
        "accuracy": mtr.make_scorer(mtr.accuracy_score),
    }
    return {s: scorers[s] for s in scorers_wanted}
