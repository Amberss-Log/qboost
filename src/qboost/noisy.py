"""Run an optimised QAOA circuit on a backend: an IBM device noise model or real hardware.

The workflow mirrors what is practical on today's devices: tune (gamma, beta) on an
ideal simulator, then execute only the final circuit on the noisy backend and measure.
The same functions accept a real IBM backend from ``QiskitRuntimeService`` unchanged.
"""

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

from qboost.qubo import QUBO
from qboost.solvers.qaoa import QAOASolver


def build_isa_circuit(
    qubo: QUBO,
    solver: QAOASolver,
    gammas: list[float],
    betas: list[float],
    backend: Any,
    seed: int = 0,
) -> QuantumCircuit:
    """Bind the angles, add measurements, and compile to the backend's native gates/layout."""
    circuit, _ = solver.build_circuit(qubo)
    bound = solver._assign(circuit, np.asarray(gammas), np.asarray(betas))
    bound.measure_all()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed)
    return pm.run(bound)


def analyse_counts(qubo: QUBO, counts: dict[str, int]) -> dict[str, Any]:
    """Score measured bitstrings against the QUBO.

    Qiskit writes bitstrings with qubit 0 on the right, so int(s, 2) is the index whose
    bit j is qubit j = QUBO variable j.
    """
    n = qubo.n
    shots = sum(counts.values())
    powers = 1 << np.arange(n)
    energies = qubo.energies((np.arange(1 << n)[:, None] & powers) > 0)
    ground = int(np.argmin(energies))
    idx = np.array([int(s, 2) for s in counts])
    freq = np.array(list(counts.values()), dtype=float) / shots
    return {
        "shots": shots,
        "p_ground_state": float(freq[idx == ground].sum()),
        "expected_energy": float(freq @ energies[idx]),
        "best_energy": float(energies[idx].min()),
        "found_optimum": bool(np.isclose(energies[idx].min(), energies[ground])),
    }


def circuit_stats(isa: QuantumCircuit) -> dict[str, int]:
    two_qubit = sum(v for k, v in isa.count_ops().items() if k in ("ecr", "cz", "cx"))
    return {"two_qubit_gates": int(two_qubit), "depth": int(isa.depth())}


def run_on_backend(
    qubo: QUBO,
    solver: QAOASolver,
    gammas: list[float],
    betas: list[float],
    backend: Any,
    shots: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    """Compile, sample once on ``backend``, and score the result."""
    isa = build_isa_circuit(qubo, solver, gammas, betas, backend, seed)
    counts: dict[str, int] = (
        SamplerV2(mode=backend).run([isa], shots=shots).result()[0].data.meas.get_counts()
    )
    return {**analyse_counts(qubo, counts), **circuit_stats(isa)}
