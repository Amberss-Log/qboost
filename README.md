# QBoost: sparse ensemble selection as a QUBO

> **Status: in progress (started Oct 2026).** Implemented: QUBO formulation, decision-stump weak
> learners, exact and simulated-annealing solvers, tests. Not yet implemented: QAOA (Qiskit),
> IBM Quantum hardware runs. No quantum results are claimed below until they exist.

## The problem

Boosting builds a strong classifier from many weak ones. **QBoost** (Neven et al., 2008) asks:
*which few weak learners should we keep?* With binary weights $w_j \in \{0,1\}$:

$$
L(w) = \frac{1}{S}\sum_{s}\Big(\frac{1}{n}\sum_j w_j h_j(x_s) - y_s\Big)^2 + \lambda \sum_j w_j
$$

Expanding the square gives a **QUBO**, $L(w) = w^\top Q w + \text{const}$ (derivation in
[`src/qboost/qubo.py`](src/qboost/qubo.py)). One binary variable per candidate learner means
**one qubit per candidate learner**, so the candidate-pool size is the qubit budget.

Application: predicting **US flight delays** (OpenML dataset 1169, derived from US Bureau of
Transportation Statistics on-time records; ~539k flights, 44.5% delayed).

## Solvers (all share one `Solver` interface and the same QUBO)

| Solver | Status | Notes |
|---|---|---|
| `ExactSolver` | done | Brute force over $2^n$; ground truth up to ~22 variables |
| `SimulatedAnnealing` | done | Metropolis single-flip chain with geometric cooling |
| QAOA (Qiskit Aer) | planned | Same QUBO mapped to an Ising Hamiltonian |
| QAOA (IBM Quantum hardware) | planned | Small instances only |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"
pytest
python scripts/run_m0.py          # downloads the dataset once, writes results/m0.md
```

## Layout

```
src/qboost/
  data.py           dataset loading
  weak.py           decision stumps
  qubo.py           QUBO container + QBoost formulation
  classifier.py     QBoostClassifier (fit stumps -> build QUBO -> solve)
  solvers/          base interface, exact, simulated annealing
tests/              pytest suite
scripts/run_m0.py   first experiment
```

## References

- H. Neven, V. Denchev, G. Rose, W. Macready, *Training a Binary Classifier with the Quantum
  Adiabatic Algorithm*, arXiv:0811.0416 (2008).
- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*,
  arXiv:1411.4028 (2014).

## License

MIT
