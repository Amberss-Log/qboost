"""M0 experiment: exact vs simulated annealing on the QBoost QUBO for flight delays.

Usage:  python scripts/run_m0.py  [--samples 20000] [--learners 12]
Writes a markdown table to results/m0.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from qboost import QBoostClassifier
from qboost.data import load_airlines
from qboost.solvers import ExactSolver, SimulatedAnnealing, Solver

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=20_000)
    p.add_argument("--learners", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, y = load_airlines(n_samples=args.samples, seed=args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    majority = max(np.mean(y_te == 1), np.mean(y_te == -1))

    rows = []
    for lam in [0.0, 0.002, 0.01]:
        solvers: list[Solver] = [ExactSolver(), SimulatedAnnealing(seed=args.seed)]
        for solver in solvers:
            t0 = time.perf_counter()
            clf = QBoostClassifier(solver, n_learners=args.learners, lam=lam).fit(X_tr, y_tr)
            secs = time.perf_counter() - t0
            acc = float(np.mean(clf.predict(X_te) == y_te))
            rows.append((lam, solver.name, clf.result_.energy, len(clf.selected_), acc, secs))

    lines = [
        f"# M0 results: {args.samples} flights, {args.learners} candidate stumps (= qubits)",
        "",
        f"Test-set majority-class baseline: {majority:.4f}",
        "",
        "| lambda | solver | QUBO energy | stumps kept | test accuracy | fit time (s) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.6f} | {r[3]} | {r[4]:.4f} | {r[5]:.2f} |" for r in rows]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
