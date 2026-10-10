"""Common solver interface: every backend takes the same QUBO and returns a SolveResult."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qboost.qubo import QUBO


@dataclass(frozen=True)
class SolveResult:
    bits: npt.NDArray[np.int8]
    energy: float
    solver: str
    info: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    name: str = "solver"

    @abstractmethod
    def solve(self, qubo: QUBO) -> SolveResult: ...
