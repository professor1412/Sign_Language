# Phase 1 execution summary

Prepared all 8,257 annotation records and their frame folders without changing source data. See `data/prepared/phase1/validation_report.json` for exact counts and fingerprints.

- Training videos retained for pilot held-out study: 2809 of 7,096.
- Selected held-out labels: 108.
- Remaining observed seen labels: 668.
- Labels incidentally losing all training examples: 309.
- Seven unit/integration tests passed, including complete synthetic bundle generation and overwrite refusal.
- CUDA was available on the RTX 4070 Laptop GPU and a small neural-layer forward pass succeeded; details are in `reports/phase1_environment.json`.

The split is seeded random pilot selection, not a completed linguistic family review. Inspect data retention and subgroup support before freezing the research protocol. No model training or CorrNet checkpoint reproduction was performed. Dictionaries are not claimed to match CorrNet checkpoint IDs. Frame-name continuity and nonzero sizes were checked for every clip; only the first PNG header per clip was validated, not full image decoding. Existing scaffold modules outside this implementation remain empty.
