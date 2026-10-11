# QBoost: sparse ensemble selection as a QUBO

> **Status: in progress (started Oct 2026).** Implemented: QUBO formulation, exact and
> simulated-annealing solvers, QAOA (Qiskit) with layer-wise warm start, noise-aware QAOA runs
> using an IBM device noise model, continuous CVQBoost, tests. **Not yet run on real quantum
> hardware.** Every result below is from a classical computer or a simulator.

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
| `QAOASolver` | done (simulator) | QUBO -> Ising Hamiltonian; COBYLA outer loop; layer-wise warm start; optional shot sampling |
| `noisy.run_on_backend` | done (noise model) | Transpiles to a device's native gates and samples with its calibrated noise; accepts a real IBM backend unchanged |
| `CVQBoostClassifier` | done | Continuous weights on the simplex (QCi's CVQBoost formulation), solved exactly as a convex QP |
| QAOA on IBM Quantum hardware | planned | Small instances only |

## Results so far

Full tables: [`results/m0.md`](results/m0.md), [`results/m1.md`](results/m1.md),
[`results/m2.md`](results/m2.md).

**Classification (20,000 flights, held-out 30%).** QBoost keeps 3 of 12 stumps and reaches
61.3% accuracy (majority baseline 55.2%); continuous CVQBoost with 40 candidates reaches 62.2%.
A one-hot logistic regression on all features reaches **63.8%, better than every QBoost
variant**: the value of QBoost here is a very sparse, interpretable ensemble and a clean
benchmark QUBO, not accuracy.

**Solver quality on the same QUBO.** Simulated annealing and QAOA recovered the exact optimum
in every case tested (4-12 variables).

**QAOA depth (ideal simulator), probability that one measurement returns the optimum:**

| qubits | random guess | p=1 | p=2 | p=3 |
|---|---|---|---|---|
| 8 | 0.0039 | 0.068 | 0.310 | 0.375 |
| 12 | 0.0002 | 0.0068 | 0.0141 | 0.0182 |

With random initial angles, p=2 was *worse* than p=1 at 8-12 qubits (e.g. 0.0018 at 8 qubits,
[`results/m1.md`](results/m1.md)): the classical optimiser was getting stuck. Initialising
depth p from the depth p-1 optimum fixed it, since that start reproduces the p-1 state exactly.

**Noise (IBM Torino noise model, QAOA p=1):** the advantage over random guessing shrinks as
circuits grow. At 8 qubits the transpiled circuit needs 92 two-qubit gates and P(optimum) falls
from 0.068 (ideal) to 0.048 (noisy), still about 12x random guessing.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[quantum,dev]"
pytest
python scripts/run_m0.py          # downloads the dataset once, writes results/m0.md
python scripts/run_m1.py          # QAOA vs exact vs annealing, writes results/m1.md
python scripts/run_m2.py          # baselines, CVQBoost, warm start, noise model (~5 min)
```

## Layout

```
src/qboost/
  data.py           dataset loading
  weak.py           decision stumps
  qubo.py           QUBO container + QBoost formulation
  ising.py          QUBO -> Ising conversion (spins = Pauli-Z eigenvalues)
  cvqboost.py       continuous CVQBoost on the simplex
  noisy.py          run a QAOA circuit on a noise model or IBM backend
  classifier.py     QBoostClassifier (fit stumps -> build QUBO -> solve)
  solvers/          base interface, exact, simulated annealing, QAOA
tests/              pytest suite
scripts/            run_m0.py, run_m1.py, run_m2.py
```

## References

- H. Neven, V. Denchev, G. Rose, W. Macready, *Training a Binary Classifier with the Quantum
  Adiabatic Algorithm*, arXiv:0811.0416 (2008).
- Quantum Computing Inc., *Profiling of CVQBoost Algorithm: Fraud Detection* (2025).
- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*,
  arXiv:1411.4028 (2014).

## License

MIT
