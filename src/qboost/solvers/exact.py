"""Brute force over all 2^n bitstrings (in chunks). Fine up to ~20 variables."""

import numpy as np

from qboost.qubo import QUBO
from qboost.solvers.base import Solver, SolveResult


class ExactSolver(Solver):
    name = "exact"

    def __init__(self, max_vars: int = 22, chunk_size: int = 1 << 16) -> None:
        self.max_vars = max_vars
        self.chunk_size = chunk_size

    def solve(self, qubo: QUBO) -> SolveResult:
        n = qubo.n
        if n > self.max_vars:
            raise ValueError(f"{n} variables exceeds max_vars={self.max_vars} for brute force.")
        powers = 1 << np.arange(n)  # bit j of the integer index is variable j
        best_energy, best_index = np.inf, 0
        for start in range(0, 1 << n, self.chunk_size):
            idx = np.arange(start, min(start + self.chunk_size, 1 << n))
            W = (idx[:, None] & powers) > 0
            E = qubo.energies(W)
            k = int(np.argmin(E))
            if E[k] < best_energy:
                best_energy, best_index = float(E[k]), int(idx[k])
        bits = ((best_index & powers) > 0).astype(np.int8)
        return SolveResult(
            bits=bits, energy=best_energy, solver=self.name, info={"evaluated": 1 << n}
        )
