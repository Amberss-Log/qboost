import itertools

import numpy as np
import pytest

from qboost.qubo import QUBO, qboost_loss, qboost_qubo


@pytest.fixture
def votes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    H = rng.choice([-1, 1], size=(200, 6)).astype(np.int8)
    y = rng.choice([-1, 1], size=200).astype(np.int8)
    return H, y


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.5])
def test_qubo_energy_equals_direct_loss_for_every_bitstring(votes, lam):
    H, y = votes
    qubo = qboost_qubo(H, y, lam)
    for bits in itertools.product([0, 1], repeat=H.shape[1]):
        assert qubo.energy(bits) == pytest.approx(qboost_loss(H, y, bits, lam))


def test_batch_energies_match_single(votes):
    H, y = votes
    qubo = qboost_qubo(H, y, 0.05)
    W = np.random.default_rng(1).integers(0, 2, size=(32, H.shape[1]))
    np.testing.assert_allclose(qubo.energies(W), [qubo.energy(w) for w in W])


def test_perfect_learner_is_selected_alone():
    y = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)
    H = np.column_stack([y, -y, np.ones_like(y)])  # perfect, anti-perfect, constant
    qubo = qboost_qubo(H, y, lam=0.0)
    # Selecting only the perfect learner gives residual (1/3)y - y; any addition is worse.
    assert np.argmin(np.diag(qubo.Q)) == 0


def test_rejects_asymmetric_or_nonsquare_q():
    with pytest.raises(ValueError):
        QUBO(Q=np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        QUBO(Q=np.zeros((2, 3)))


def test_rejects_mismatched_labels(votes):
    H, y = votes
    with pytest.raises(ValueError):
        qboost_qubo(H, y[:-1], 0.0)
