# Phase 2 implementation and verification

Date: 2026-11-23 (Asia/Calcutta).

## What is implemented

- A CorrNet ResNet18 adapter using the official source modules, strict checkpoint loading, and a separate released vocabulary.
- Deterministic image preprocessing, TemporalConv feature extraction, disk caching, extraction resume, checksums and cache-equivalence checks.
- The custom shared temporal encoder and linear head A, with padding-aware convolutions and packed BiLSTM sequences.
- Validated CTC loss, correct repeated-label decoding, corpus WER with substitution/deletion/insertion counts.
- Small-set overfitting, training, dev-based checkpoint selection, epoch-boundary resume, and saved-model evaluation.
- Beginner commands and explanations in [docs/phase2.md](../docs/phase2.md).

## Checks actually performed

| Check | Result |
|---|---|
| Phase 1 and Phase 2 automated tests | 24 passed in the final validation run |
| Synthetic feature learning on the NVIDIA GPU | Four of four sequences learned in 188 updates, including repeated labels |
| Synthetic loss | 5.1307 initially; 0.0661 at the last training update |
| Synthetic checkpoint reload | Same output logits after reload |
| Full-size custom encoder | Forward and backward pass passed |
| Padding invariance | Adding another, longer video to a batch preserves valid predictions |
| CPU training resume | Three-epoch uninterrupted and interrupted/resumed runs produced exactly equal weights and histories |
| GPU command workflow | Default-size model trained one epoch, resumed to epoch two, and produced an evaluation report on synthetic data |
| Small-set overfit command | Learned all four synthetic training sequences in 188 updates; training-set WER 0% |
| Missing checkpoint handling | Clear error, with no output cache created |
| Real PNG preprocessing + CorrNet architecture | Passed for a 16-frame PHOENIX dev clip using explicitly untrained weights |
| Untrained CorrNet cache boundary | Serialized feature route produced identical logits; maximum absolute difference 0 |

The real-image architecture check used video `11December_2009_Friday_tagesschau-3509`. Its 16 frames became 28 after reference padding and produced features shaped `(1, 4, 1024)`. Peak allocated GPU memory in that check was 502,801,408 bytes. This tiny-clip measurement is not a bound for long videos.

Local detailed artifacts:

- `runs/phase2_smoke/smoke_report.json`
- `runs/phase2_adapter_check/report.json`
- `runs/phase2_cli_check/history.json`
- `runs/phase2_cli_check/evaluation.json`
- `runs/phase2_overfit_cli/overfit_report.json`

The GPU command workflow intentionally stopped after two training updates. Its synthetic evaluation WER was 100%; this was a command/resume check, not a trained recognition baseline. The separate synthetic overfit check above learned all four sequences.

## Reference assets

Official source: [hulianyuyy/CorrNet](https://github.com/hulianyuyy/CorrNet), checked out at `a812814f4deefda0b75ab1dcc90b106c62912263` in `external/corrnet/upstream`.

The released PHOENIX-2014-T dictionary contains 1,115 glosses plus blank. It was converted to `artifacts/pretrained/corrnet/vocabulary.json`. The custom Phase 1 training dictionary remains separate, with 1,085 glosses plus blank. No original dataset files were modified.

## What is still pending

1. Download the authors' PHOENIX-2014-T checkpoint to `artifacts/pretrained/corrnet/phoenix2014t.pt`.
2. Run trained reference predictions and serialized-feature equivalence checks on real clips.
3. Extract the full feature caches and overfit a handful of real training clips.
4. Train head A on the full training cache, choose the checkpoint using dev WER, and evaluate the final test split.
5. For an exact published-reference comparison, match the upstream beam decoder and scoring procedure. The implemented baseline uses greedy decoding and raw-gloss WER.

The automatic approval review could not complete the checkpoint-download request because its service reported a usage limit. That download was not executed. This was a review-service failure, not a finding that the download was unsafe.

**No pretrained CorrNet WER or full-data head-A WER has been measured here.** The published reference values in the plan remain reference numbers. Phase 2's implementation is available, but its full experimental completion criteria remain unmet until the real-data runs above finish.
