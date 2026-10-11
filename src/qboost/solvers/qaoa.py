"""QAOA for a QUBO with Qiskit (Farhi et al. 2014).

Each layer is exp(-i gamma H_C) followed by exp(-i beta sum X). COBYLA tunes the
angles to minimise <H_C>, then the circuit is sampled and the best bitstring kept.

shots=None uses exact statevector probabilities. shots=k estimates them from k
shots per evaluation instead (closer to real hardware).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from qboost.ising import Ising, qubo_to_ising
from qboost.qubo import QUBO
from qboost.solvers.base import Solver, SolveResult

FloatArray = npt.NDArray[np.float64]


def ising_operator(ising: Ising) -> SparsePauliOp:
    """Ising cost operator in Pauli-Z terms (constant dropped). Qubit j = variable j."""
    terms: list[tuple[str, list[int], float]] = [
        ("Z", [j], float(ising.h[j])) for j in range(ising.n) if ising.h[j] != 0.0
    ]
    rows, cols = np.nonzero(ising.J)
    terms += [
        ("ZZ", [int(j), int(k)], float(ising.J[j, k])) for j, k in zip(rows, cols, strict=True)
    ]
    if not terms:  # all-zero problem, still need a valid operator
        terms = [("Z", [0], 0.0)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=ising.n)


class QAOASolver(Solver):
    name = "qaoa"

    def __init__(
        self,
        reps: int = 1,
        shots: int | None = None,
        n_starts: int = 4,
        maxiter: int = 200,
        final_shots: int = 1024,
        max_qubits: int = 16,
        seed: int | None = None,
        warm_start: bool = True,
    ) -> None:
        self.reps = reps
        self.shots = shots
        self.n_starts = n_starts
        self.maxiter = maxiter
        self.final_shots = final_shots
        self.max_qubits = max_qubits
        self.seed = seed
        self.warm_start = warm_start

    def build_circuit(self, qubo: QUBO) -> tuple[QuantumCircuit, float]:
        """QAOA ansatz for the QUBO.

        H_C is scaled so its largest coefficient is 1. The optimum doesn't change,
        but gamma stays in a sensible range for any problem.
        """
        op = ising_operator(qubo_to_ising(qubo))
        scale = float(np.max(np.abs(op.coeffs))) or 1.0
        return qaoa_ansatz((op / scale).simplify(), reps=self.reps), scale

    def _assign(
        self, circuit: QuantumCircuit, gammas: FloatArray, betas: FloatArray
    ) -> QuantumCircuit:
        # qaoa_ansatz names its parameters β[i] and γ[i].
        values: dict[Any, float] = {}
        for p in circuit.parameters:
            idx = int(p.name.split("[")[1].rstrip("]"))
            values[p] = float(gammas[idx] if p.name.startswith("γ") else betas[idx])
        return circuit.assign_parameters(values)

    def _probabilities(
        self, circuit: QuantumCircuit, shots: int | None, rng: np.random.Generator
    ) -> FloatArray:
        """Probability of each bitstring index, exact or estimated from shots."""
        probs: FloatArray = Statevector(circuit).probabilities()
        if shots is None:
            return probs
        counts = rng.multinomial(shots, probs / probs.sum())
        return counts.astype(float) / shots

    def solve(self, qubo: QUBO) -> SolveResult:
        n = qubo.n
        if n > self.max_qubits:
            raise ValueError(f"{n} qubits exceeds max_qubits={self.max_qubits} for simulation.")
        rng = np.random.default_rng(self.seed)
        circuit, _ = self.build_circuit(qubo)

        # energy of every bitstring, same order as Statevector.probabilities()
        powers = 1 << np.arange(n)
        all_bits = (np.arange(1 << n)[:, None] & powers) > 0
        energies = qubo.energies(all_bits)

        n_evals = 0

        def expected_energy(theta: FloatArray) -> float:
            nonlocal n_evals
            n_evals += 1
            bound = self._assign(circuit, theta[: self.reps], theta[self.reps :])
            return float(self._probabilities(bound, self.shots, rng) @ energies)

        starts = [
            np.concatenate([rng.uniform(0, 2 * np.pi, self.reps), rng.uniform(0, np.pi, self.reps)])
            for _ in range(self.n_starts)
        ]
        if self.warm_start and self.reps > 1:
            # warm start: solve depth p-1 first and start depth p from those angles
            # with the new layer at 0. That point is exactly the p-1 state, so p
            # can't end up worse.
            prev = QAOASolver(
                reps=self.reps - 1,
                shots=self.shots,
                n_starts=self.n_starts,
                maxiter=self.maxiter,
                final_shots=self.final_shots,
                max_qubits=self.max_qubits,
                seed=self.seed,
                warm_start=True,
            ).solve(qubo)
            starts.insert(0, np.array(prev.info["gammas"] + [0.0] + prev.info["betas"] + [0.0]))

        best: Any = None
        for theta0 in starts:
            res = minimize(
                expected_energy, theta0, method="COBYLA", options={"maxiter": self.maxiter}
            )
            if best is None or res.fun < best.fun:
                best = res

        # sample the final circuit and keep the best bitstring seen
        theta = best.x
        bound = self._assign(circuit, theta[: self.reps], theta[self.reps :])
        exact_probs = self._probabilities(bound, None, rng)
        samples = rng.choice(1 << n, size=self.final_shots, p=exact_probs / exact_probs.sum())
        winner = int(samples[np.argmin(energies[samples])])
        ground = int(np.argmin(energies))

        bits = ((winner & powers) > 0).astype(np.int8)
        return SolveResult(
            bits=bits,
            energy=float(energies[winner]),
            solver=f"{self.name}_p{self.reps}",
            info={
                "gammas": theta[: self.reps].tolist(),
                "betas": theta[self.reps :].tolist(),
                "expected_energy": float(exact_probs @ energies),
                "p_ground_state": float(exact_probs[ground]),
                "p_ground_state_uniform": 1.0 / (1 << n),
                "circuit_evaluations": n_evals,
            },
        )
