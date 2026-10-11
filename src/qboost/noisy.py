"""Run an optimised QAOA circuit on a backend: an IBM device noise model or real hardware.

The workflow mirrors what is practical on today's devices: tune (gamma, beta) on an
ideal simulator, then execute only the final circuit on the noisy backend and measure.
The same function accepts a real IBM backend from ``QiskitRuntimeService`` unchanged.
"""

from typing import Any

import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2

from qboost.qubo import QUBO
from qboost.solvers.qaoa import QAOASolver


def run_on_backend(
    qubo: QUBO,
    solver: QAOASolver,
    gammas: list[float],
    betas: list[float],
    backend: Any,
    shots: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    circuit, _ = solver.build_circuit(qubo)
    bound = solver._assign(circuit, np.asarray(gammas), np.asarray(betas))
    bound.measure_all()
    isa = transpile(bound, backend=backend, optimization_level=3, seed_transpiler=seed)

    job = SamplerV2(mode=backend).run([isa], shots=shots)
    counts: dict[str, int] = job.result()[0].data.meas.get_counts()

    n = qubo.n
    powers = 1 << np.arange(n)
    energies = qubo.energies((np.arange(1 << n)[:, None] & powers) > 0)
    ground = int(np.argmin(energies))

    # Qiskit bitstrings are written with qubit 0 on the right, so int(s, 2) is the index
    # whose bit j is qubit j = QUBO variable j.
    idx = np.array([int(s, 2) for s in counts])
    freq = np.array(list(counts.values()), dtype=float) / shots
    return {
        "p_ground_state": float(freq[idx == ground].sum()),
        "expected_energy": float(freq @ energies[idx]),
        "best_energy": float(energies[idx].min()),
        "found_optimum": bool(np.isclose(energies[idx].min(), energies[ground])),
        "two_qubit_gates": int(
            sum(v for k, v in isa.count_ops().items() if k in ("ecr", "cz", "cx"))
        ),
        "depth": int(isa.depth()),
    }
