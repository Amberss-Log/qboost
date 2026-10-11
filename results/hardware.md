# Real-hardware results

- Backend: **ibm_kingston**
- Job id: `db5l8tcvf2bc73d03bfg`
- Date (UTC): 2026-10-11 10:48
- Shots per circuit: 4096
- QPU time: 4.0 s

QAOA p=1 on the QBoost flight-delay QUBO (20,000 flights). Angles optimised on the
ideal simulator; the compiled circuits were then sampled on the device and on its
noise model. P(opt) = fraction of shots returning the optimal ensemble.

| qubits | random guess | P(opt) ideal | P(opt) noise model | **P(opt) real QPU** | optimum seen on QPU | 2-qubit gates | depth |
|---|---|---|---|---|---|---|---|
| 4 | 0.0625 | 0.1502 | 0.1499 | **0.1624** | True | 18 | 64 |
| 6 | 0.0156 | 0.1078 | 0.0986 | **0.0942** | True | 53 | 147 |
