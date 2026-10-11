# QBoost: Quantum Optimization for Sparse Ensemble Learning

This project formulates weak-learner selection in boosting (QBoost) as a quadratic
unconstrained binary optimization (QUBO) problem and compares classical and quantum
solvers on it: exhaustive search, simulated annealing, and the Quantum Approximate
Optimization Algorithm (QAOA) implemented in Qiskit. The application is flight-delay
prediction on US airline data.

**Status:** in progress. All results below were obtained on classical hardware or
quantum simulators. Runs on IBM Quantum hardware are in progress.

## Contents

- [Method](#method)
- [Implementation](#implementation)
- [Results](#results)
- [Usage](#usage)
- [Project structure](#project-structure)
- [References](#references)

## Method

### QBoost formulation

Given $n$ candidate weak learners $h_j(x) \in \{-1, +1\}$ and $S$ training samples with
labels $y_s \in \{-1, +1\}$, QBoost selects a subset of learners through binary weights
$w_j \in \{0, 1\}$ by minimizing

$$
L(w) = \frac{1}{S}\sum_{s=1}^{S}\Big(\frac{1}{n}\sum_{j=1}^{n} w_j h_j(x_s) - y_s\Big)^2 + \lambda \sum_{j=1}^{n} w_j .
$$

The first term measures how well the ensemble vote matches the labels; the second
penalizes ensemble size. Expanding the square and using $w_j^2 = w_j$ gives a QUBO,
$L(w) = w^\top Q w + c$, with

$$
Q = \frac{H^\top H}{n^2 S}, \qquad Q_{jj} \mathrel{+}= -\frac{2\,(H^\top y)_j}{nS} + \lambda ,
$$

where $H_{sj} = h_j(x_s)$. Each candidate learner corresponds to one binary variable,
and therefore to one qubit. The off-diagonal terms of $Q$ penalize selecting learners
that vote alike, so the solution favours small, non-redundant ensembles.

### Data and weak learners

- **Dataset:** OpenML Airlines (ID 1169), derived from US Bureau of Transportation
  Statistics on-time records: 539,383 flights, 44.5% delayed. Experiments use a random
  subset of 20,000 flights with a 70/30 train/test split.
- **Weak learners:** decision stumps on single features (departure-time thresholds,
  airline, origin and destination airport). The most accurate $n$ stumps form the
  candidate pool.

### Solvers

| Solver | Description |
|---|---|
| Exact | Exhaustive search over all $2^n$ assignments; reference optimum for $n \le 20$. |
| Simulated annealing | Single-bit-flip Metropolis moves with a geometric cooling schedule and random restarts. |
| QAOA | QUBO mapped to an Ising Hamiltonian; circuit angles optimized with COBYLA; depth-$p$ circuits initialized from the optimized depth-$(p-1)$ angles. |
| CVQBoost | Continuous relaxation with weights on the probability simplex ($w_j \ge 0$, $\sum_j w_j = 1$), following QCi's CVQBoost formulation; solved as a convex quadratic program. |

## Implementation

- All solvers implement a common `Solver` interface: they take a `QUBO` object and return
  a `SolveResult` containing the selected bitstring and its energy.
- QAOA can be evaluated with exact statevector probabilities or with finite measurement
  shots.
- Noise studies transpile the optimized circuit to a device's native gate set and
  connectivity, then sample it under the device's calibrated noise model
  (`qiskit_ibm_runtime` fake backends). The same code path accepts a real IBM backend.
- The package is covered by a pytest suite (40 tests), type-checked with mypy and linted
  with ruff.

## Results

Full tables are in [`results/`](results/).

### Classification accuracy

Held-out test set, 6,000 flights.

| Model | Learners used | Test accuracy |
|---|---|---|
| Majority class | – | 55.2% |
| QBoost (exact, annealing and QAOA select the same ensemble) | 3 of 12 | 61.3% |
| CVQBoost | 10 of 40 | 62.2% |
| Logistic regression (all features, one-hot encoded) | – | 63.8% |

QBoost produces a very sparse ensemble but does not outperform logistic regression on
this dataset. Its main use here is as a realistic benchmark problem for comparing
optimization methods.

### Solver comparison

Simulated annealing and QAOA recovered the exact optimum on every instance tested
(4 to 12 variables).

### QAOA circuit depth

Probability that a single measurement returns the optimal ensemble (ideal simulator):

| Qubits | Random guess | $p = 1$ | $p = 2$ | $p = 3$ |
|---|---|---|---|---|
| 8 | 0.0039 | 0.068 | 0.310 | 0.375 |
| 12 | 0.0002 | 0.0068 | 0.0141 | 0.0182 |

With randomly initialized angles, depth 2 initially performed worse than depth 1
(0.0018 at 8 qubits; see [`results/m1.md`](results/m1.md)). Since a depth-2 circuit can
represent any depth-1 state, this pointed to the classical optimizer converging to a
poor local minimum. Initializing depth $p$ from the optimized depth-$(p-1)$ angles, with
the added layer set to zero, resolved the issue.

### Effect of device noise

QAOA ($p = 1$) sampled under the IBM Torino noise model:

| Qubits | Random guess | Ideal | Noisy | Two-qubit gates | Circuit depth |
|---|---|---|---|---|---|
| 4 | 0.0625 | 0.150 | 0.153 | 16 | 58 |
| 6 | 0.0156 | 0.108 | 0.091 | 46 | 122 |
| 8 | 0.0039 | 0.068 | 0.048 | 92 | 198 |

The two-qubit gate count grows quickly with problem size because of limited qubit
connectivity, and the success probability degrades accordingly. At 8 qubits it remains
about 12 times higher than random guessing.

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux/macOS
pip install -e ".[quantum,dev]"
pytest
```

| Script | Purpose |
|---|---|
| `scripts/run_m0.py` | Exact search vs simulated annealing |
| `scripts/run_m1.py` | QAOA vs problem size |
| `scripts/run_m2.py` | Baselines, CVQBoost, QAOA depth and noise study (about 5 minutes) |
| `scripts/run_hardware.py` | QAOA on IBM Quantum hardware (`--dry-run` uses a simulated backend) |

To run on IBM Quantum hardware, create a `.env` file in the repository root containing
`IBM_QUANTUM_TOKEN=<your token>`. The file is excluded from version control.

## Project structure

```
src/qboost/
  data.py          dataset loading
  weak.py          decision-stump weak learners
  qubo.py          QUBO representation and QBoost formulation
  ising.py         QUBO to Ising conversion
  classifier.py    QBoost classifier
  cvqboost.py      continuous (simplex-constrained) variant
  noisy.py         execution on simulated and real IBM backends
  ibm.py           IBM Quantum authentication
  solvers/         exact, simulated annealing and QAOA solvers
scripts/           experiment scripts
tests/             test suite
results/           experiment outputs
```

## References

1. H. Neven, V. S. Denchev, G. Rose and W. G. Macready, "Training a Binary Classifier
   with the Quantum Adiabatic Algorithm," arXiv:0811.0416, 2008.
2. E. Farhi, J. Goldstone and S. Gutmann, "A Quantum Approximate Optimization Algorithm,"
   arXiv:1411.4028, 2014.
3. Quantum Computing Inc., "Profiling of CVQBoost Algorithm: Fraud Detection," 2025.

## License

MIT
