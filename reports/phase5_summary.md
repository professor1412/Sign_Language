# Phase 5 implementation and available evidence

Recorded 2026-11-25.

The Phase 5 reporting, model export and cached-clip demonstration commands are implemented. Final real-data research results depend on completing the training and evaluation runs in Phases 2–4.

## Delivered functionality

- `final-report` creates CSV, JSON and Markdown summaries from saved evaluations. It verifies scores against predictions and includes correct examples, frequent confusions, missing signs and extra signs.
- `export-model` packages trained recognition weights, vocabulary, configuration and feature-producer identity. It supports older checkpoints that store the producer hash alone and preserves expanded candidate metadata.
- `predict` reloads a checkpoint or export, checks cache compatibility and predicts one cached clip. Prediction depends on the saved visual features. Reference glosses are available for evaluation separately.
- New evaluations record their split. Original evaluation metadata, including unseen metrics and candidate labels, is carried into final reports.

## Verification

All 45 Phase 1–5 tests passed. The five Phase 5 tests check A/B/C export parity, older checkpoint compatibility, wrong feature producers, missing clips, error accounting and rejection of inconsistent metrics. A command-line demonstration successfully exported and reloaded the existing synthetic Head B checkpoint. Its prediction and diagnostic comparison report are in `runs/phase5_checks/`.

## Available real-data measurement

The earlier released-CorrNet check evaluated five development clips with 35 reference glosses. It counted three substitutions, one deletion and three insertions, giving 20% WER with greedy decoding and raw annotation scoring. The local report at `reports/final/corrnet_five_clip/report.md` records these examples and their source metadata. The original evaluation predates the explicit data-origin and split fields, so its generated summary marks those fields as unspecified. The input path identifies the five-clip development check.

## Pending research deliverables

| Deliverable | Current evidence |
|---|---|
| Full Head A dev and test results | Pending training and final evaluation |
| Real-data A/B/C and control comparison | Pending training and matched evaluations |
| Repeated-seed variation | Pending real runs |
| Clean unseen-sign results | Pending reduced-data backbone and head training |
| Final explanations of model strengths and limitations | Pending those measured results |
| Saved real-model demonstration | Requires a trained real-data head checkpoint |

Spiking models and the whole-video summary experiment are deferred optional extensions. The implemented demonstration uses a completed feature cache. Raw video upload, live camera input and the service API remain future work.

The [Phase 5 guide](../docs/phase5.md) gives the commands for producing the final artifacts once the real checkpoints are ready.
