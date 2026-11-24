# Phase 5 — Results and a saved-clip demonstration

Phase 5 uses the checkpoints and predictions produced in Phases 2–4. The commands below create reports and a portable recognizer for a completed experiment. Final scientific conclusions require the real training and evaluation runs.

## Generate a report

Choose the model and settings using the allowed development split, then run the existing evaluation commands on the final test set. Pass their JSON reports to this command.

```powershell
$env:PYTHONPATH = 'src'
python -m sign_language.cli.main final-report --evaluations runs/head_a_seed42/dev_evaluation.json runs/head_a_seed42/test_evaluation.json --output reports/final/head_a
```

The new directory contains `results.csv`, `results.json` and `report.md`. The command recomputes corpus WER from saved predictions and checks it against recorded metrics. It includes correct examples, missing signs, extra signs, substitutions and frequent confusions. The JSON keeps checkpoint hashes, evaluation subsets, candidate lists and unseen-sign statistics whenever the source reports supply them. Earlier evaluations without a split field show an unspecified split.

Use `compare-runs` for matching training settings and seed sets across A, B, C and controls. The final report presents individual evaluations with their source metadata. Full-data recognition and reduced-data unseen-sign conclusions each need their own experiment evidence.

```powershell
python -m sign_language.cli.main compare-runs --runs runs/known_signs/head_a_seed42 runs/known_signs/head_b_seed42 runs/known_signs/head_c_seed42 --output reports/tables/known_sign_comparison.json
```

Those run folders must first complete their configured epoch budgets. Repeat seeds where practical and state whether the backbone or only the heads were retrained. Describe optional spiking and whole-video-summary experiments as deferred unless their implementation and evaluation are completed.

## Export the chosen recognizer

```powershell
python -m sign_language.cli.main export-model --checkpoint runs/head_a_seed42/best.pt --output artifacts/exports/head_a.pt
```

The export contains the temporal encoder, head weights, vocabulary, model settings and feature-producer identity. B/C exports also carry the text vectors stored in the trained model. Exported checkpoints support prediction with the matching cached features. Keep the recorded CorrNet checkpoint and preprocessing available when generating further features. Use the original training checkpoint for resume.

## Demonstrate one clip

```powershell
python -m sign_language.cli.main predict --checkpoint artifacts/exports/head_a.pt --cache data/processed/corrnet_reference/dev --clip-id 11August_2010_Wednesday_tagesschau-2 --output reports/final/head_a_prediction.json --device cuda
```

This loads one clip from a completed feature cache and prints its predicted glosses. The cache identity must match the feature producer used during training. The demonstration uses saved features and supports A/B/C and expanded text-head checkpoints. Arbitrary video upload, live camera input and a service API remain future work.

On the current Windows environment, the previously verified PowerShell helper disables cuDNN while retaining CUDA. Define it in the same terminal and use it for prediction or evaluation if the cuDNN shutdown issue occurs.

```powershell
function Invoke-Phase2Cuda {
    python -c 'import torch; torch.backends.cudnn.enabled = False; from sign_language.cli.main import main; main()' @args
}
Invoke-Phase2Cuda predict --checkpoint artifacts/exports/head_a.pt --cache data/processed/corrnet_reference/dev --clip-id 11August_2010_Wednesday_tagesschau-2 --output reports/final/head_a_prediction_cuda.json --device cuda
```

Use a fresh output path for each report, export or prediction. Each command refuses an existing output.

## Completion evidence

The project report should include the fixed split and candidate labels, training settings and seed counts, dev and test scores, correct and incorrect predictions, limitations, and a reproducible saved-model example. Keep published benchmark scores and locally measured results clearly attributed. Current measurements and pending experiments are recorded in `reports/phase5_summary.md`.
