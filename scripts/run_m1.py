"""QAOA (ideal sim) vs exact vs annealing for 4-12 qubits -> results/m1.md

python scripts/run_m1.py [--samples 20000] [--lam 0.0]
"""

import argparse
import time
from pathlib import Path

from qboost.data import load_airlines
from qboost.qubo import qboost_qubo
from qboost.solvers import ExactSolver, SimulatedAnnealing, Solver
from qboost.solvers.qaoa import QAOASolver
from qboost.weak import fit_stumps, predict_matrix

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=20_000)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, y = load_airlines(n_samples=args.samples, seed=args.seed)

    rows = []
    for n in [4, 6, 8, 10, 12]:
        stumps = fit_stumps(X, y, n_learners=n)
        qubo = qboost_qubo(predict_matrix(stumps, X), y, args.lam)
        ground = ExactSolver().solve(qubo).energy
        solvers: list[Solver] = [
            SimulatedAnnealing(seed=args.seed),
            QAOASolver(reps=1, seed=args.seed),
            QAOASolver(reps=2, seed=args.seed),
        ]
        for solver in solvers:
            t0 = time.perf_counter()
            r = solver.solve(qubo)
            secs = time.perf_counter() - t0
            p_gs = r.info.get("p_ground_state")
            rows.append(
                (
                    n,
                    r.solver,
                    r.energy - ground,
                    "-" if p_gs is None else f"{p_gs:.4f}",
                    f"{1 / 2**n:.4f}",
                    secs,
                )
            )

    lines = [
        f"# M1 results: {args.samples} flights, lambda = {args.lam}, ideal (noiseless) simulator",
        "",
        "Energy gap = solver energy - exact optimum (0 means the optimum was found).",
        "P(optimum) = probability one QAOA measurement returns the optimal bitstring;",
        "compare with 1/2^n for random guessing.",
        "",
        "| qubits | solver | energy gap | P(optimum) | random guess 1/2^n | time (s) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.2e} | {r[3]} | {r[4]} | {r[5]:.1f} |" for r in rows]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
