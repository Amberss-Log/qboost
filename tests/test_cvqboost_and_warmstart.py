import numpy as np
import pandas as pd
import pytest

from qboost.cvqboost import CVQBoostClassifier, SimplexQP, cvqboost_problem, solve_simplex_qp
from qboost.qubo import QUBO


def test_simplex_solution_is_feasible_and_beats_vertices_and_centre():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    qp = SimplexQP(J=A @ A.T + 0.1 * np.eye(6), C=rng.normal(size=6))
    w = solve_simplex_qp(qp)
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)
    candidates = [np.eye(6)[i] for i in range(6)] + [np.full(6, 1 / 6)]
    assert all(qp.objective(w) <= qp.objective(c) + 1e-9 for c in candidates)


def test_cvqboost_objective_matches_direct_loss():
    rng = np.random.default_rng(1)
    H = rng.choice([-1, 1], size=(50, 4))
    y = rng.choice([-1, 1], size=50)
    w = rng.dirichlet(np.ones(4))
    qp = cvqboost_problem(H, y, lam=0.3)
    direct = np.mean((H @ w - y) ** 2) - 1.0 + 0.3 * np.sum(w**2)  # constant mean(y^2)=1 dropped
    assert qp.objective(w) == pytest.approx(direct)


def test_cvqboost_classifier_puts_weight_on_the_informative_feature():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"good": rng.integers(0, 100, 500), "noise": rng.integers(0, 100, 500)})
    y = np.where(X["good"] > 50, 1, -1).astype(np.int8)
    clf = CVQBoostClassifier(n_learners=6).fit(X, y)
    assert np.mean(clf.predict(X) == y) > 0.9
    assert {s.feature for s in clf.selected_} >= {"good"}


qiskit = pytest.importorskip("qiskit")
from qboost.solvers.qaoa import QAOASolver  # noqa: E402


def test_warm_started_depth_two_is_no_worse_than_depth_one():
    A = np.random.default_rng(7).normal(size=(6, 6))
    qubo = QUBO(Q=(A + A.T) / 2)
    p1 = QAOASolver(reps=1, seed=0).solve(qubo).info["expected_energy"]
    p2 = QAOASolver(reps=2, seed=0, warm_start=True).solve(qubo).info["expected_energy"]
    assert p2 <= p1 + 1e-9
