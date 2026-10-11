"""M2 experiment: the full comparison.

A. Classification on held-out flights: majority class, logistic regression, QBoost
   (exact / simulated annealing / QAOA), and continuous CVQBoost.
B. QAOA depth with layer-wise warm start (ideal simulator).
C. QAOA under a real IBM device noise model (FakeTorino, 133-qubit Heron), no hardware.

Usage:  python scripts/run_m2.py
Writes results/m2.md.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit_ibm_runtime.fake_provider import FakeTorino
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from qboost import QBoostClassifier
from qboost.cvqboost import CVQBoostClassifier
from qboost.data import load_airlines
from qboost.noisy import run_on_backend
from qboost.qubo import qboost_qubo
from qboost.solvers import ExactSolver, SimulatedAnnealing
from qboost.solvers.qaoa import QAOASolver
from qboost.weak import fit_stumps, predict_matrix

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 0


def classification(lines: list[str]) -> None:
    X, y = load_airlines(n_samples=20_000, seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    rows: list[tuple[str, str, str]] = []

    majority = max(np.mean(y_te == 1), np.mean(y_te == -1))
    rows.append(("Majority class", "-", f"{majority:.4f}"))

    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    logreg = make_pipeline(
        ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
                ("num", StandardScaler(), num),
            ]
        ),
        LogisticRegression(max_iter=2000),
    ).fit(X_tr, y_tr)
    rows.append(
        (
            "Logistic regression (all features, one-hot)",
            "-",
            f"{np.mean(logreg.predict(X_te) == y_te):.4f}",
        )
    )

    for label, solver in [
        ("QBoost, exact solver", ExactSolver()),
        ("QBoost, simulated annealing", SimulatedAnnealing(seed=SEED)),
        ("QBoost, QAOA p=1 (ideal simulator)", QAOASolver(reps=1, seed=SEED)),
    ]:
        clf = QBoostClassifier(solver, n_learners=12).fit(X_tr, y_tr)
        rows.append(
            (label, f"{len(clf.selected_)} of 12", f"{np.mean(clf.predict(X_te) == y_te):.4f}")
        )

    for n in (12, 40):
        cv = CVQBoostClassifier(n_learners=n).fit(X_tr, y_tr)
        rows.append(
            (
                f"CVQBoost, continuous weights ({n} candidates)",
                f"{len(cv.selected_)} of {n}",
                f"{np.mean(cv.predict(X_te) == y_te):.4f}",
            )
        )

    lines += [
        "## A. Flight-delay classification (20,000 flights, 70/30 split)",
        "",
        "| model | weak learners used | test accuracy |",
        "|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines.append("")


def depth_study(lines: list[str]) -> None:
    X, y = load_airlines(n_samples=20_000, seed=SEED)
    lines += [
        "## B. QAOA depth with layer-wise warm start (ideal simulator)",
        "",
        "P(optimum) = probability that one measurement returns the optimal ensemble.",
        "",
        "| qubits | random guess | p=1 | p=2 (warm start) | p=3 (warm start) |",
        "|---|---|---|---|---|",
    ]
    for n in (8, 12):
        qubo = qboost_qubo(predict_matrix(fit_stumps(X, y, n), X), y, 0.0)
        ps = [QAOASolver(reps=p, seed=SEED).solve(qubo).info["p_ground_state"] for p in (1, 2, 3)]
        lines.append(f"| {n} | {1 / 2**n:.4f} | " + " | ".join(f"{v:.4f}" for v in ps) + " |")
    lines.append("")


def noise_study(lines: list[str]) -> None:
    X, y = load_airlines(n_samples=20_000, seed=SEED)
    backend = FakeTorino()
    lines += [
        "## C. QAOA p=1 under IBM Torino's noise model (simulated, not real hardware)",
        "",
        "Angles optimised on the ideal simulator; the final circuit is transpiled to the",
        "device's native gates and connectivity and sampled with its calibrated noise.",
        "",
        "| qubits | random guess | P(opt) ideal | P(opt) noisy | optimum seen (noisy) "
        "| 2-qubit gates | depth |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in (4, 6, 8):
        qubo = qboost_qubo(predict_matrix(fit_stumps(X, y, n), X), y, 0.0)
        solver = QAOASolver(reps=1, seed=SEED)
        ideal = solver.solve(qubo).info
        noisy = run_on_backend(
            qubo, solver, ideal["gammas"], ideal["betas"], backend, shots=4096, seed=SEED
        )
        lines.append(
            f"| {n} | {1 / 2**n:.4f} | {ideal['p_ground_state']:.4f} | "
            f"{noisy['p_ground_state']:.4f} | {noisy['found_optimum']} | "
            f"{noisy['two_qubit_gates']} | {noisy['depth']} |"
        )
    lines.append("")


def main() -> None:
    t0 = time.perf_counter()
    lines = ["# M2 results", ""]
    classification(lines)
    depth_study(lines)
    noise_study(lines)
    lines.append(f"_Total runtime: {time.perf_counter() - t0:.0f} s._")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
