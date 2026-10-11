import itertools

import numpy as np
import pytest

from qboost.ising import qubo_to_ising
from qboost.qubo import QUBO
from qboost.solvers import ExactSolver

qiskit = pytest.importorskip("qiskit")

from qboost.solvers.qaoa import QAOASolver, ising_operator  # noqa: E402


def random_qubo(n: int, seed: int) -> QUBO:
    A = np.random.default_rng(seed).normal(size=(n, n))
    return QUBO(Q=(A + A.T) / 2, offset=0.7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ising_energy_matches_qubo_for_every_bitstring(seed):
    qubo = random_qubo(5, seed)
    ising = qubo_to_ising(qubo)
    for w in itertools.product([0, 1], repeat=5):
        z = 1 - 2 * np.array(w)
        assert ising.energy(z) == pytest.approx(qubo.energy(w))


def test_operator_diagonal_matches_qubo_with_qiskit_bit_order():
    """Checks qubit j <-> variable j: the operator's diagonal, indexed the way Qiskit
    indexes basis states, must equal the QUBO energies (up to the dropped constant)."""
    qubo = random_qubo(4, 3)
    ising = qubo_to_ising(qubo)
    diag = np.real(ising_operator(ising).to_matrix().diagonal())
    powers = 1 << np.arange(4)
    W = (np.arange(16)[:, None] & powers) > 0
    np.testing.assert_allclose(diag + ising.const, qubo.energies(W), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_qaoa_finds_ground_state_on_small_problem(seed):
    qubo = random_qubo(4, seed)
    result = QAOASolver(reps=2, seed=seed).solve(qubo)
    assert result.energy == pytest.approx(ExactSolver().solve(qubo).energy)


def test_qaoa_beats_uniform_random_guessing():
    qubo = random_qubo(6, 4)
    info = QAOASolver(reps=2, seed=0).solve(qubo).info
    mean_uniform = float(np.mean(qubo.energies((np.arange(64)[:, None] & (1 << np.arange(6))) > 0)))
    assert info["expected_energy"] < mean_uniform
    assert info["p_ground_state"] > info["p_ground_state_uniform"]


def test_qaoa_with_shot_noise_runs():
    result = QAOASolver(reps=1, shots=256, n_starts=1, seed=0).solve(random_qubo(4, 5))
    assert result.bits.shape == (4,)


def test_qaoa_refuses_too_many_qubits():
    with pytest.raises(ValueError):
        QAOASolver(max_qubits=3).solve(random_qubo(4, 0))
