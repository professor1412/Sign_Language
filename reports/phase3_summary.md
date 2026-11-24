# Phase 3 implementation and verification

Recorded on 2026-11-24. Phase numbers follow the workspace five-phase plan.

Code is implemented for audited German fastText embedding preparation, heads B/C, A-norm, B-rand, B-shuf, shared training/checkpoint/evaluation, and controlled dev-run summaries. Real PHOENIX experiments remain pending.

## Checks completed

- All 31 Phase 1-3 unit/integration tests passed with `PYTHONPATH=src`.
- All six heads trained for three debug epochs and their checkpoints reloaded and evaluated through the shared command implementation.
- B learned all four invented sequences in 60 CPU updates; C in 65. Both reached zero training WER, including repeated-gloss sequences.
- Text vectors stay fixed while video projections, special vectors and C's text projection receive gradients.
- All heads start from identical encoder weights for the same seed.
- Interrupted/resumed C training matches uninterrupted CPU model state and history exactly, including when the source embedding bundle is removed before resume.
- Vocabulary mismatches, zero vectors, incompatible comparisons, incomplete training budgets and unequal seed sets are checked.

Artifacts: [smoke report](../runs/phase3_smoke/smoke_report.json), [debug run comparison](../runs/phase3_smoke/comparison.json), [B learning check](../runs/phase3_smoke/b_overfit/overfit_report.json), [C learning check](../runs/phase3_smoke/c_overfit/overfit_report.json).

Dummy train/dev patterns intentionally repeat. Dummy word vectors are orthogonal invented vectors, not fastText semantics. The three-epoch comparison only tests execution/reporting and is not a ranking of the methods. These results do not measure real PHOENIX accuracy or show that language meaning helps recognition.

## Pending real experiment

1. Obtain the released CorrNet checkpoint and extract the real train/dev feature caches using Phase 2.
2. Install the optional fastText loader, obtain German `cc.de.300.bin`, generate the real embedding bundle and inspect its audit.
3. Run a small real-clip learning check, then full comparable A/B/C/control experiments with matching seeds.
4. Repeat important comparisons, inspect dev WER/blank rate/recall, freeze choices, and finally evaluate test.

No real fastText binary load or real-data Phase 3 training has been performed. The fastText export orchestration is tested using a mocked loader. Greedy decoding is supported; beam decoding, clean held-out vocabulary expansion and deployment remain later work. See [the Phase 3 guide](../docs/phase3.md) for commands.
