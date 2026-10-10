import itertools

import numpy as np
import pytest

from qboost.qubo import QUBO
from qboost.solvers import ExactSolver, SimulatedAnnealing
from qboost.solvers.anneal import metropolis_accept


def random_qubo(n: int, seed: int) -> QUBO:
    A = np.random.default_rng(seed).normal(size=(n, n))
    return QUBO(Q=(A + A.T) / 2, offset=0.3)


def brute_force_min(qubo: QUBO) -> float:
    return min(qubo.energy(b) for b in itertools.product([0, 1], repeat=qubo.n))


@pytest.mark.parametrize("n,seed", [(1, 0), (5, 1), (9, 2)])
def test_exact_matches_itertools(n, seed):
    qubo = random_qubo(n, seed)
    result = ExactSolver(chunk_size=7).solve(qubo)  # small chunks exercise the chunk loop
    assert result.energy == pytest.approx(brute_force_min(qubo))
    assert qubo.energy(result.bits) == pytest.approx(result.energy)


def test_exact_refuses_too_many_variables():
    with pytest.raises(ValueError):
        ExactSolver(max_vars=4).solve(random_qubo(5, 0))


def test_metropolis_always_accepts_downhill_and_flat_moves():
    rng = np.random.default_rng(0)
    assert all(metropolis_accept(d, 1.0, rng) for d in [-5.0, -1e-9, 0.0])


def test_metropolis_uphill_rate_matches_boltzmann_factor():
    rng = np.random.default_rng(0)
    delta, T, trials = 1.0, 2.0, 20_000
    rate = np.mean([metropolis_accept(delta, T, rng) for _ in range(trials)])
    assert rate == pytest.approx(np.exp(-delta / T), abs=0.015)


def test_metropolis_handles_huge_uphill_without_overflow():
    rng = np.random.default_rng(0)
    assert metropolis_accept(1e6, 1e-6, rng) is False


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_annealing_finds_ground_state_on_small_problems(seed):
    qubo = random_qubo(10, seed)
    result = SimulatedAnnealing(n_sweeps=300, n_restarts=4, seed=seed).solve(qubo)
    assert result.energy == pytest.approx(ExactSolver().solve(qubo).energy)
    assert 0.0 < result.info["acceptance_rate"] < 1.0
