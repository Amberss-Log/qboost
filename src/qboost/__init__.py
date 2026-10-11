"""QBoost ensemble selection written as a QUBO, plus a few solvers for it."""

from qboost.classifier import QBoostClassifier
from qboost.qubo import QUBO, qboost_qubo

__all__ = ["QBoostClassifier", "QUBO", "qboost_qubo"]
__version__ = "0.1.0"
