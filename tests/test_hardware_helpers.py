from pathlib import Path

import numpy as np
import pytest

from qboost.ibm import get_service, load_env
from qboost.qubo import QUBO

pytest.importorskip("qiskit")
from qboost.noisy import analyse_counts  # noqa: E402


def test_load_env_parses_comments_blanks_and_quotes(tmp_path: Path):
    f = tmp_path / ".env"
    f.write_text(
        "# comment\n\n"
        "IBM_QUANTUM_TOKEN='abc123'\n"
        "IBM_QUANTUM_INSTANCE=crn:x  # trailing\n"
        "BAD LINE\n",
        encoding="utf-8",
    )
    assert load_env(f) == {"IBM_QUANTUM_TOKEN": "abc123", "IBM_QUANTUM_INSTANCE": "crn:x"}


def test_load_env_missing_file_is_empty(tmp_path: Path):
    assert load_env(tmp_path / "nope.env") == {}


def test_get_service_without_token_fails_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="IBM_QUANTUM_TOKEN"):
        get_service(tmp_path / "empty.env")


def test_analyse_counts_uses_qiskit_bit_order():
    # Q rewards variable 0 only: optimum is w = (1, 0) -> index 1 -> Qiskit string "01".
    qubo = QUBO(Q=np.array([[-1.0, 0.0], [0.0, 1.0]]))
    out = analyse_counts(qubo, {"01": 30, "10": 50, "00": 20})
    assert out["shots"] == 100
    assert out["p_ground_state"] == pytest.approx(0.30)
    assert out["found_optimum"] is True
    assert out["expected_energy"] == pytest.approx(0.30 * -1 + 0.50 * 1 + 0.20 * 0)
