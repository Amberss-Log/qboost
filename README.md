# QBoost: sparse ensemble selection as a QUBO

> **Status: in progress (started Oct 2026).** Implemented: QUBO formulation, decision-stump weak
> learners, exact and simulated-annealing solvers, QAOA on an ideal (noiseless) Qiskit simulator,
> tests. Not yet done: noisy simulation, IBM Quantum hardware runs. Every result below is from a
> classical computer or a noiseless simulator.

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
| `QAOASolver` | done (ideal simulator) | QUBO -> Ising Hamiltonian; COBYLA outer loop; optional shot sampling |
| QAOA on IBM Quantum hardware | planned | Small instances only |

## Results so far

- [`results/m0.md`](results/m0.md): 20k flights, 12 candidate stumps. Simulated annealing matches
  the exact optimum at every lambda tested; QBoost keeps 2-3 of 12 stumps and reaches 61.3% test
  accuracy vs a 55.2% majority-class baseline (no stronger classical baseline compared yet).
- [`results/m1.md`](results/m1.md): QAOA on the same QUBO, 4-12 qubits, noiseless simulator.
  Depth p=1 raises the probability of measuring the optimal ensemble above random guessing at
  every size (e.g. 0.0068 vs 0.0002 at 12 qubits). Depth p=2 was *worse* than p=1 at 8-12 qubits
  with random initial angles, which points to the classical optimiser getting stuck, not to a
  limit of the circuit (p=2 contains p=1). Next: warm-start p=2 from the p=1 optimum.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[quantum,dev]"
pytest
python scripts/run_m0.py          # downloads the dataset once, writes results/m0.md
python scripts/run_m1.py          # QAOA vs exact vs annealing, writes results/m1.md
```

## Layout

```
src/qboost/
  data.py           dataset loading
  weak.py           decision stumps
  qubo.py           QUBO container + QBoost formulation
  ising.py          QUBO -> Ising conversion (spins = Pauli-Z eigenvalues)
  classifier.py     QBoostClassifier (fit stumps -> build QUBO -> solve)
  solvers/          base interface, exact, simulated annealing, QAOA
tests/              pytest suite
scripts/            run_m0.py (classical), run_m1.py (QAOA)
```

## References

- H. Neven, V. Denchev, G. Rose, W. Macready, *Training a Binary Classifier with the Quantum
  Adiabatic Algorithm*, arXiv:0811.0416 (2008).
- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*,
  arXiv:1411.4028 (2014).

## License

MIT
