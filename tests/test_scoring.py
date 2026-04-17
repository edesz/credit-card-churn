#!/usr/bin/env python3


"""Test prediction scorers."""


import numpy as np
import pytest

import src.cc_churn.scoring as sc


def test_get_scorers_keys():
    """Test get_scorers with valid keys."""
    scorers = sc.get_scorers(["f2", "recall", "accuracy"])
    assert set(scorers.keys()) == {"f2", "recall", "accuracy"}


def test_get_scorers_invalid_key():
    """Test get_scorers with invalid key."""
    with pytest.raises(KeyError):
        sc.get_scorers(["invalid_metric"])


def test_get_scorers_callable():
    """Test get_scorers for a callable."""
    scorers = sc.get_scorers(["f1"])
    scorer = scorers["f1"]
    assert callable(scorer)


def test_scorer_execution(binary_data):
    """Test scorer fu nctionality."""
    y_true, y_pred, _ = binary_data
    scorers = sc.get_scorers(["accuracy", "recall", "f1"])
    results = {
        k: scorer._score_func(y_true, y_pred, **scorer._kwargs)
        for k, scorer in scorers.items()
    }
    assert np.isclose(results["accuracy"], 0.5)
    assert np.isclose(results["recall"], 0.5)
    assert np.isclose(results["f1"], 0.5)


def test_prauc_scorer(binary_data):
    """Test prauc behavior."""
    y_true, y_pred, _ = binary_data
    scorers = sc.get_scorers(["prauc"])
    result = scorers["prauc"]._score_func(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_rocauc_scorer(binary_data):
    y_true, _, y_pred_proba = binary_data
    scorers = sc.get_scorers(["rocauc"])
    result = scorers["rocauc"]._score_func(y_true, y_pred_proba)
    assert 0.0 <= result <= 1.0
