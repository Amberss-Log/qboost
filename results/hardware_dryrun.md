# Hardware dry run (simulated)

- Backend: **FakeTorino (dry run, simulated)**
- Job id: `f437ca03-1bdd-4d7c-bed2-b5fcbbe86047`
- Date (UTC): 2026-10-11 09:13
- Shots per circuit: 4096
- QPU time: not reported

QAOA p=1 on the QBoost flight-delay QUBO (20,000 flights). Angles optimised on the
ideal simulator; the compiled circuits were then sampled on the device and on its
noise model. P(opt) = fraction of shots returning the optimal ensemble.

| qubits | random guess | P(opt) ideal | P(opt) noise model | **P(opt) real QPU** | optimum seen on QPU | 2-qubit gates | depth |
|---|---|---|---|---|---|---|---|
| 4 | 0.0625 | 0.1502 | 0.1504 | **0.1484** | True | 16 | 58 |
| 6 | 0.0156 | 0.1078 | 0.0854 | **0.0889** | True | 46 | 122 |
