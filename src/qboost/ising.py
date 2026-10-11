"""QUBO -> Ising conversion.

Substituting w_j = (1 - z_j) / 2 with spins z_j in {+1, -1} turns a QUBO into

    E(z) = const + sum_j h_j z_j + sum_{j<k} J_jk z_j z_k.

Quantum hardware measures the Pauli-Z operator, whose eigenvalues are exactly these
spins: qubit state |0> has z = +1 (w = 0) and |1> has z = -1 (w = 1). So the
bitstring read out of a qubit register *is* the QUBO assignment w.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qboost.qubo import QUBO

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Ising:
    h: FloatArray  # shape (n,), local fields
    J: FloatArray  # shape (n, n), strictly upper triangular couplings
    const: float

    @property
    def n(self) -> int:
        return int(self.h.shape[0])

    def energy(self, z: npt.ArrayLike) -> float:
        s = np.asarray(z, dtype=float)
        return float(self.const + self.h @ s + s @ self.J @ s)


def qubo_to_ising(qubo: QUBO) -> Ising:
    """Exact conversion; E_ising(1 - 2w) == E_qubo(w) for every bitstring w."""
    Q = qubo.Q
    diag = np.diag(Q)
    off = Q - np.diag(diag)  # symmetric, zero diagonal
    # E = sum_j Q_jj w_j + sum_{j<k} 2 Q_jk w_j w_k + offset, then substitute w = (1 - z)/2.
    h = -diag / 2.0 - off.sum(axis=1) / 2.0
    J = np.triu(off, k=1) / 2.0
    const = qubo.offset + diag.sum() / 2.0 + np.triu(off, k=1).sum() / 2.0
    return Ising(h=h, J=J, const=float(const))
