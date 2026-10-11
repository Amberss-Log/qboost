# qboost

Picking a small set of weak classifiers with QBoost, written as a QUBO and solved with
brute force, simulated annealing and QAOA (Qiskit). Work in progress. So far everything has
run on a laptop or a simulator; real IBM hardware runs are in progress.

## Idea

QBoost (Neven et al., 2008) chooses which weak learners to keep in a boosted ensemble.
Each learner gets a weight $w_j \in \{0,1\}$ and the loss is

$$
L(w) = \frac{1}{S}\sum_{s}\Big(\frac{1}{n}\sum_j w_j h_j(x_s) - y_s\Big)^2 + \lambda \sum_j w_j
$$

Expanding the square gives a QUBO, $L(w) = w^\top Q w + \text{const}$ (see
[`qubo.py`](src/qboost/qubo.py)). One variable per learner means one qubit per learner, so
the number of candidate learners is the number of qubits.

The weak learners are decision stumps, and the data is the OpenML airline dataset (id 1169,
from US BTS on-time records, about 539k flights, 44.5% delayed). Most runs use a random
20k subset.

## What's in here

- `ExactSolver`: brute force over all $2^n$ bitstrings, used as ground truth (up to ~20 variables)
- `SimulatedAnnealing`: single bit-flip Metropolis with geometric cooling
- `QAOASolver`: QUBO -> Ising -> QAOA in Qiskit, angles tuned with COBYLA, deeper circuits
  warm-started from the shallower ones
- `noisy.py`: runs the final QAOA circuit on a fake IBM backend (noise model) or a real one
- `CVQBoostClassifier`: the continuous version (weights on the simplex, as in QCi's CVQBoost),
  solved with scipy

All solvers take the same `QUBO` object and return a `SolveResult`.

## Results so far

Tables are in [`results/`](results/).

Accuracy on held-out flights (20k flights, 70/30 split):

| model | test accuracy |
|---|---|
| majority class | 55.2% |
| QBoost, 12 candidate stumps (exact, SA and QAOA all pick the same 3) | 61.3% |
| CVQBoost, 40 candidate stumps | 62.2% |
| logistic regression, all features one-hot | 63.8% |

Logistic regression is still better. What QBoost gives here is a very small ensemble
(3 stumps) and a decent test problem for comparing solvers, not better accuracy.

Annealing and QAOA found the exact optimum in every case I ran (4-12 variables).

Probability that a single QAOA measurement gives the optimal answer (ideal simulator):

| qubits | random guess | p=1 | p=2 | p=3 |
|---|---|---|---|---|
| 8 | 0.0039 | 0.068 | 0.310 | 0.375 |
| 12 | 0.0002 | 0.0068 | 0.0141 | 0.0182 |

My first p=2 runs started from random angles and came out worse than p=1 (0.0018 at
8 qubits, see `results/m1.md`), which shouldn't happen since p=2 includes p=1. Starting
p=2 from the best p=1 angles, with the extra layer at zero, fixed it.

With the FakeTorino noise model (p=1), 8 qubits needs 92 two-qubit gates after
transpiling and the success probability drops from 0.068 to 0.048. That's still about
12x random guessing, but the drop gets bigger as the circuits grow.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[quantum,dev]"
pytest
python scripts/run_m0.py        # exact vs annealing
python scripts/run_m1.py        # QAOA vs qubit count
python scripts/run_m2.py        # baselines, CVQBoost, depth, noise (~5 min)
python scripts/run_hardware.py --dry-run   # hardware script on FakeTorino
```

For real hardware, put `IBM_QUANTUM_TOKEN=...` in a `.env` file in the repo root and run
`python scripts/run_hardware.py`.

## Files

```
src/qboost/
  data.py         loads the airline data
  weak.py         decision stumps
  qubo.py         QUBO class + QBoost loss
  ising.py        QUBO -> Ising
  classifier.py   QBoostClassifier
  cvqboost.py     continuous version
  noisy.py        running circuits on fake/real IBM backends
  ibm.py          IBM Quantum login from .env
  solvers/        exact, annealing, QAOA
scripts/          experiments
tests/
results/
```

## References

- H. Neven, V. Denchev, G. Rose, W. Macready. Training a Binary Classifier with the Quantum
  Adiabatic Algorithm. arXiv:0811.0416 (2008).
- E. Farhi, J. Goldstone, S. Gutmann. A Quantum Approximate Optimization Algorithm.
  arXiv:1411.4028 (2014).
- Quantum Computing Inc. Profiling of CVQBoost Algorithm: Fraud Detection (2025).

MIT license.
