"""QBoost: sparse ensemble selection as a QUBO, with pluggable classical and quantum solvers."""

from qboost.classifier import QBoostClassifier
from qboost.qubo import QUBO, qboost_qubo

__all__ = ["QBoostClassifier", "QUBO", "qboost_qubo"]
__version__ = "0.1.0"
