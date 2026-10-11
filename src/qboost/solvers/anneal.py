"""Simulated annealing for QUBOs.

Single bit-flip Metropolis moves with geometric cooling. At fixed T this samples
p(w) ~ exp(-E(w)/T). The proposal is symmetric, so there is no Hastings correction.
"""

import numpy as np
import numpy.typing as npt

from qboost.qubo import QUBO
from qboost.solvers.base import Solver, SolveResult


def metropolis_accept(delta: float, temperature: float, rng: np.random.Generator) -> bool:
    """Metropolis rule: accept downhill moves, otherwise accept with prob exp(-delta/T).

    delta <= 0 is checked first so the exponent is always negative (no overflow).
    """
    if delta <= 0.0:
        return True
    return bool(rng.random() < np.exp(-delta / temperature))


class SimulatedAnnealing(Solver):
    name = "simulated_annealing"

    def __init__(
        self,
        n_sweeps: int = 500,
        n_restarts: int = 8,
        t_start: float | None = None,
        t_end: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_sweeps = n_sweeps
        self.n_restarts = n_restarts
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed

    def schedule(self, qubo: QUBO) -> npt.NDArray[np.float64]:
        """Geometric schedule. Default T0 = largest possible |dE| for a single flip."""
        Q = qubo.Q
        diag = np.abs(np.diag(Q))
        max_delta = float(np.max(diag + 2.0 * (np.abs(Q).sum(axis=1) - diag)))
        t0 = self.t_start if self.t_start is not None else max(max_delta, 1e-12)
        t1 = self.t_end if self.t_end is not None else t0 * 1e-3
        return np.geomspace(t0, t1, self.n_sweeps)

    def solve(self, qubo: QUBO) -> SolveResult:
        rng = np.random.default_rng(self.seed)
        Q, n = qubo.Q, qubo.n
        temps = self.schedule(qubo)
        best_bits, best_energy = np.zeros(n), np.inf
        accepted = proposed = 0

        for _ in range(self.n_restarts):
            w = rng.integers(0, 2, size=n).astype(float)
            local_field = Q @ w  # local_field[j] = sum_k Q[j, k] w[k]
            energy = qubo.energy(w)
            if energy < best_energy:
                best_energy, best_bits = energy, w.copy()
            for T in temps:
                for j in rng.permutation(n):
                    d = 1.0 - 2.0 * w[j]  # +1 for a 0->1 flip, -1 for 1->0
                    delta = d * (Q[j, j] + 2.0 * (local_field[j] - Q[j, j] * w[j]))
                    proposed += 1
                    if metropolis_accept(float(delta), float(T), rng):
                        accepted += 1
                        w[j] += d
                        local_field += d * Q[:, j]
                        energy += delta
                        if energy < best_energy:
                            best_energy, best_bits = energy, w.copy()

        return SolveResult(
            bits=best_bits.astype(np.int8),
            # recompute, the running energy drifts slightly
            energy=qubo.energy(best_bits),
            solver=self.name,
            info={"acceptance_rate": accepted / max(proposed, 1)},
        )
