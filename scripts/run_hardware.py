"""QAOA (p=1) on a real IBM device, 4 and 6 qubits, same data as run_m2.

Angles come from the ideal simulator; both circuits go to the QPU as one job and
are also run on that backend's noise model to compare.

    python scripts/run_hardware.py --dry-run       # FakeTorino, no token needed
    python scripts/run_hardware.py                 # real device (token in .env)
    python scripts/run_hardware.py --job-id <id>   # get results of an earlier job
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from qboost.data import load_airlines
from qboost.noisy import analyse_counts, build_isa_circuit, circuit_stats
from qboost.qubo import QUBO, qboost_qubo
from qboost.solvers.qaoa import QAOASolver
from qboost.weak import fit_stumps, predict_matrix

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 0


def build_problems(sizes: list[int]) -> list[dict[str, Any]]:
    X, y = load_airlines(n_samples=20_000, seed=SEED)
    problems = []
    for n in sizes:
        qubo = qboost_qubo(predict_matrix(fit_stumps(X, y, n), X), y, 0.0)
        solver = QAOASolver(reps=1, seed=SEED)
        info = solver.solve(qubo).info
        problems.append({"n": n, "qubo": qubo, "solver": solver, "ideal": info})
    return problems


def counts_of(result: Any, i: int) -> dict[str, int]:
    return dict(result[i].data.meas.get_counts())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--job-id")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--wait-minutes", type=float, default=30.0)
    args = ap.parse_args()

    problems = build_problems(args.sizes)

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        backend: Any = FakeTorino()
        job = None
    else:
        from qboost.ibm import get_service

        service = get_service()
        if args.job_id:
            job = service.job(args.job_id)
            backend = job.backend()
        else:
            job = None
            backend = service.least_busy(
                operational=True, simulator=False, min_num_qubits=max(args.sizes)
            )
    print(f"Backend: {backend.name}")

    isas = [
        build_isa_circuit(
            p["qubo"], p["solver"], p["ideal"]["gammas"], p["ideal"]["betas"], backend, seed=SEED
        )
        for p in problems
    ]

    if job is None:
        job = SamplerV2(mode=backend).run(isas, shots=args.shots)
    job_id = job.job_id()
    print(f"Job id: {job_id}  (re-fetch later with --job-id {job_id})")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "hardware_job.json").write_text(
        json.dumps({"job_id": job_id, "backend": backend.name, "dry_run": args.dry_run}, indent=2),
        encoding="utf-8",
    )

    if not args.dry_run:
        deadline = time.time() + 60 * args.wait_minutes
        status = str(job.status())
        while status not in ("DONE", "ERROR", "CANCELLED") and time.time() < deadline:
            print(f"  status: {status} ... waiting")
            time.sleep(20)
            status = str(job.status())
        if status != "DONE":
            print(f"Job not finished (status {status}). Re-run with --job-id {job_id} later.")
            return
    hw = job.result()

    # same circuits on the backend's noise model, locally
    sim = AerSimulator.from_backend(backend)
    sim_res = SamplerV2(mode=sim).run(isas, shots=args.shots).result()

    qpu_seconds: float | None = None
    if not args.dry_run:
        try:
            qpu_seconds = float(job.usage())
        except Exception:  # usage reporting differs between plans/versions
            qpu_seconds = None

    label = "FakeTorino (dry run, simulated)" if args.dry_run else backend.name
    lines = [
        "# Real-hardware results" if not args.dry_run else "# Hardware dry run (simulated)",
        "",
        f"- Backend: **{label}**",
        f"- Job id: `{job_id}`",
        f"- Date (UTC): {datetime.now(UTC):%Y-%m-%d %H:%M}",
        f"- Shots per circuit: {args.shots}",
        f"- QPU time: {qpu_seconds:.1f} s"
        if qpu_seconds is not None
        else "- QPU time: not reported",
        "",
        "QAOA p=1 on the QBoost flight-delay QUBO (20,000 flights). Angles optimised on the",
        "ideal simulator; the compiled circuits were then sampled on the device and on its",
        "noise model. P(opt) = fraction of shots returning the optimal ensemble.",
        "",
        "| qubits | random guess | P(opt) ideal | P(opt) noise model | **P(opt) real QPU** "
        "| optimum seen on QPU | 2-qubit gates | depth |",
        "|---|---|---|---|---|---|---|---|",
    ]
    raw: dict[str, Any] = {"job_id": job_id, "backend": label, "circuits": []}
    for i, p in enumerate(problems):
        qubo: QUBO = p["qubo"]
        real = analyse_counts(qubo, counts_of(hw, i))
        model = analyse_counts(qubo, counts_of(sim_res, i))
        stats = circuit_stats(isas[i])
        lines.append(
            f"| {p['n']} | {1 / 2 ** p['n']:.4f} | {p['ideal']['p_ground_state']:.4f} | "
            f"{model['p_ground_state']:.4f} | **{real['p_ground_state']:.4f}** | "
            f"{real['found_optimum']} | {stats['two_qubit_gates']} | {stats['depth']} |"
        )
        raw["circuits"].append(
            {
                "qubits": p["n"],
                "gammas": p["ideal"]["gammas"],
                "betas": p["ideal"]["betas"],
                "counts_hardware": counts_of(hw, i),
                "counts_noise_model": counts_of(sim_res, i),
            }
        )

    name = "hardware_dryrun" if args.dry_run else "hardware"
    (RESULTS / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / f"{name}_counts.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
