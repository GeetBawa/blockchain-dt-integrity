# Blockchain for digital-twin integrity and provenance - experiment code

Reproduces every number in the chapter's experimental section.

**Data**: NASA C-MAPSS turbofan degradation, subset FD001 (`data/`), real data. Cite the original source:
A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository.
Tampering is *simulated by code* (`src/pipeline.py`); the sensor data itself is not synthetic.

**Chain**: py-evm (via eth-tester) local Ethereum test chain, Solidity 0.8.26, cancun EVM, optimiser 200 runs.
Gas figures are real EVM gas. Wall-clock times are local-process times and exclude network and consensus delay.

## Run order
```
node src/compile.js                  # compile contracts/TwinProvenanceRegistry.sol -> build/
python3 src/model.py 0 1 2 3 4       # train 5 LSTM twin models (seeds)
python3 src/exp_detect.py all        # E6 security tests + E1 detection (10 seeds)
python3 src/exp_cost.py 100 50 20 10 5 1   # E2 gas, E3 latency, E4 provenance, E7 storage (run on an idle machine)
python3 src/exp_impact.py            # E5 twin accuracy under attack
python3 src/make_report.py           # tables + figures
```
Python deps: torch, scikit-learn, pandas, numpy, matplotlib, web3[tester]. Node dep: solc@0.8.26 (npm).
`src/compile.js` has an absolute path to the solc install; edit it for your machine.

## Files
- `contracts/` Solidity: `TwinProvenanceRegistry` (full) and `LeanRegistry`
- `src/pipeline.py` batching, Merkle anchoring, verification, attacks, statistical baseline
- `results/` raw CSV/JSON outputs, tables, trained models
- `figures/` 300 dpi PNG figures
