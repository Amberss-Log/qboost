"""QAOA (Farhi, Goldstone & Gutmann, 2014) for a QUBO, using Qiskit.

The circuit starts in the uniform superposition over all 2^n bitstrings and applies
``reps`` alternating layers:

    cost layer   exp(-i * gamma * H_C)   adds an energy-dependent phase to each bitstring
    mixer layer  exp(-i * beta * sum X)  lets amplitude flow between neighbouring bitstrings

A classical optimiser (COBYLA) tunes (gamma, beta) to minimise the expected energy
<H_C>, then the circuit is sampled and the lowest-energy bitstring seen is returned.

Two evaluation modes:
* ``shots=None``: exact statevector probabilities (ideal, noiseless simulator).
* ``shots=k``: k measurement samples per evaluation (finite sampling, like hardware).
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
    """Cost Hamiltonian as Pauli-Z terms. Qubit j <-> QUBO variable j. Constant dropped."""
    terms: list[tuple[str, list[int], float]] = [
        ("Z", [j], float(ising.h[j])) for j in range(ising.n) if ising.h[j] != 0.0
    ]
    rows, cols = np.nonzero(ising.J)
    terms += [
        ("ZZ", [int(j), int(k)], float(ising.J[j, k])) for j, k in zip(rows, cols, strict=True)
    ]
    if not terms:  # degenerate all-zero problem: keep a valid operator
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
    ) -> None:
        self.reps = reps
        self.shots = shots
        self.n_starts = n_starts
        self.maxiter = maxiter
        self.final_shots = final_shots
        self.max_qubits = max_qubits
        self.seed = seed

    def build_circuit(self, qubo: QUBO) -> tuple[QuantumCircuit, float]:
        """QAOA circuit for the QUBO, with H_C rescaled to max |coefficient| = 1.

        Rescaling does not change which bitstring is optimal, but keeps good gamma
        values in a fixed range (0, 2*pi) whatever the problem's units.
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
        """Probability of each bitstring index (bit j of the index = qubit j = variable j)."""
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

        # Energy of every bitstring, in the same index order as Statevector.probabilities().
        powers = 1 << np.arange(n)
        all_bits = (np.arange(1 << n)[:, None] & powers) > 0
        energies = qubo.energies(all_bits)

        n_evals = 0

        def expected_energy(theta: FloatArray) -> float:
            nonlocal n_evals
            n_evals += 1
            bound = self._assign(circuit, theta[: self.reps], theta[self.reps :])
            return float(self._probabilities(bound, self.shots, rng) @ energies)

        best: Any = None
        for _ in range(self.n_starts):
            theta0 = np.concatenate(
                [rng.uniform(0, 2 * np.pi, self.reps), rng.uniform(0, np.pi, self.reps)]
            )
            res = minimize(
                expected_energy, theta0, method="COBYLA", options={"maxiter": self.maxiter}
            )
            if best is None or res.fun < best.fun:
                best = res

        # Final readout: sample the optimised circuit, keep the best bitstring observed.
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
