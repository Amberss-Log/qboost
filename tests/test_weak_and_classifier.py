import numpy as np
import pandas as pd
import pytest

from qboost import QBoostClassifier
from qboost.solvers import ExactSolver
from qboost.weak import Stump, candidate_stumps, fit_stumps, predict_matrix


@pytest.fixture
def toy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "time": rng.integers(0, 2400, n),
            "airline": pd.Categorical(rng.choice(["AA", "WN", "DL"], n)),
        }
    )
    # Late flights on WN are delayed; label noise keeps it from being trivial.
    signal = (X["time"] > 1500) | (X["airline"] == "WN")
    noise = rng.random(n) < 0.1
    y = np.where(signal ^ noise, 1, -1).astype(np.int8)
    return X, y


def test_stump_requires_exactly_one_condition():
    with pytest.raises(ValueError):
        Stump(feature="x")
    with pytest.raises(ValueError):
        Stump(feature="x", threshold=1.0, category="a")


def test_stump_votes_and_polarity(toy):
    X, _ = toy
    s = Stump(feature="airline", category="WN")
    np.testing.assert_array_equal(s.predict(X), np.where(X["airline"] == "WN", 1, -1))
    flipped = Stump(feature="airline", category="WN", polarity=-1)
    np.testing.assert_array_equal(flipped.predict(X), -s.predict(X))


def test_candidates_cover_numeric_and_categorical(toy):
    X, _ = toy
    features = {s.feature for s in candidate_stumps(X, n_thresholds=4)}
    assert features == {"time", "airline"}


def test_fit_stumps_are_all_better_than_chance(toy):
    X, y = toy
    stumps = fit_stumps(X, y, n_learners=5)
    H = predict_matrix(stumps, X)
    assert H.shape == (len(X), 5)
    assert np.all((H == y[:, None]).mean(axis=0) >= 0.5)


def test_classifier_beats_majority_baseline(toy):
    X, y = toy
    clf = QBoostClassifier(ExactSolver(), n_learners=8, lam=0.0).fit(X, y)
    majority = max(np.mean(y == 1), np.mean(y == -1))
    assert np.mean(clf.predict(X) == y) > majority
    assert 1 <= len(clf.selected_) <= 8


def test_large_lambda_selects_nothing_and_predicts_majority(toy):
    X, y = toy
    clf = QBoostClassifier(ExactSolver(), n_learners=6, lam=10.0).fit(X, y)
    assert clf.selected_ == []
    assert np.all(clf.predict(X) == clf.tie_label_)
