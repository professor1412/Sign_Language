# Phase 2: your first recognition model

The Phase 2 code is implemented. It can read videos through CorrNet, save features, train your own sequence model and head A, and score predictions.

**The full research phase is not finished yet.** The pretrained CorrNet checkpoint has not been downloaded. Real feature extraction, a small overfit run on real clips, full training, and a measured PHOENIX baseline remain to be run. The checks already performed are in [the results note](../reports/phase2_summary.md).

## 1. Understand the flow

```text
Your existing PNG images
  -> frozen CorrNet visual encoder and TemporalConv
  -> saved features: one file per video, with 1024 numbers per time step
  -> your trainable temporal encoder
  -> head A: scores for each gloss and blank
  -> CTC decoding
  -> predicted gloss sequence
```

The first model turns pictures into useful numbers. Saving those numbers means later training does not need to open every image again. A **gloss** is a sign label, such as WETTER. This phase predicts gloss sequences, not translated German sentences.

Two models are involved:

| Model | Purpose | Dictionary |
|---|---|---|
| Released CorrNet | Check the existing reference and extract features | Authors' original 1,115 gloss IDs plus blank |
| Your custom model with head A | Establish the baseline for your later ideas | Phase 1's 1,085 training gloss IDs plus blank |

Their label IDs differ. Never substitute one dictionary for the other. Evaluation retains unfamiliar reference labels and counts their errors; it does not silently delete them.

## 2. Open PowerShell in this project

Run commands from `C:\Users\Manish\Desktop\Sign_Language`:

```powershell
$env:PYTHONPATH = 'src'
python -m sign_language.cli.main --help
```

Your current Python already has the required packages, including CUDA-enabled PyTorch. In a new environment, install the project and Phase 2 dependencies with:

```powershell
python -m pip install -e ".[phase2]"
```

The Python packages are PyTorch, NumPy, OpenCV, and PyYAML. Phase 1 still works without them. After installation you can also use `sign-language` instead of `python -m sign_language.cli.main`.

## 3. Run a small check before using real videos

```powershell
python -m sign_language.cli.main smoke-test --output runs/my_phase2_check
```

This creates four **invented feature sequences**, teaches a small model their answers, and checks the saved model can be reloaded. It includes repeated labels such as `A A`, which require correct CTC handling. By default it uses your GPU if available. Add `--device cpu` to use the CPU.

Open `runs/my_phase2_check/smoke_report.json`. Both `overfit_passed` and `checkpoint_reload_passed` should be `true`.

This is a plumbing check, not evidence that the model understands real sign language. Its synthetic dev examples repeat the invented training patterns deliberately; their score is not a generalization result. An existing output folder is protected from accidental overwriting, so use a different folder for a new check.

## 4. Prepare the existing CorrNet model (plan step 2.1)

The official source is already downloaded into `external/corrnet/upstream`. The adapter was checked against commit `a812814f4deefda0b75ab1dcc90b106c62912263`.

For a fresh project copy, obtain that source with:

```powershell
git clone https://github.com/hulianyuyy/CorrNet.git external/corrnet/upstream
git -C external/corrnet/upstream checkout a812814f4deefda0b75ab1dcc90b106c62912263
```

Download the **PHOENIX2014-T** checkpoint from the [official CorrNet results table](https://github.com/hulianyuyy/CorrNet#inference), using its [Google Drive link](https://drive.google.com/file/d/1c_wNHYMqCbqRE5KqrQL1P6chOw5VBS6Q/view). Save the actual downloaded weights as:

```text
artifacts/pretrained/corrnet/phoenix2014t.pt
```

The `.pt` filename is our local convention; renaming the downloaded weight file does not change its contents. Do not save a Google Drive HTML page as a checkpoint. The similarly named PHOENIX2014 checkpoint is for a different dataset.

The matching dictionary has already been converted to `artifacts/pretrained/corrnet/vocabulary.json`. In a fresh copy, create it with:

```powershell
python -m sign_language.cli.main import-corrnet-vocabulary --trusted-npy external/corrnet/upstream/preprocess/phoenix2014-T/gloss_dict.npy
```

The original NumPy dictionary uses pickle, so this importer is explicitly for the trusted authors' file. Checkpoint loading uses PyTorch's restricted `weights_only=True` mode by default. If the authors' checkpoint includes old pickled training metadata that this mode rejects, use `--trust-legacy-checkpoint` only for a file whose origin you have independently trusted.

The adapter builds the upstream ResNet18, TemporalConv, BiLSTM and normalized classifier, then loads all model weights strictly. It does not download ImageNet weights or require `ctcdecode`.

## 5. Extract five videos first (plan steps 2.2–2.4)

After the checkpoint is present:

```powershell
python -m sign_language.cli.main extract-features --manifest data/prepared/phase1/manifests/reference/dev.jsonl --output data/processed/corrnet_reference/dev_first5 --limit 5
```

For each video, the command:

1. Reads its images in order, including your flat extracted folder layout.
2. Resizes each image to 256 × 256 with OpenCV Lanczos, then takes the center 224 × 224 crop. This follows the upstream resize and evaluation path.
3. Converts pixels to RGB values between -1 and 1.
4. Repeats boundary frames using CorrNet's temporal padding rule. It never randomly drops or flips frames.
5. Runs the frozen reference model in evaluation mode, without gradients.
6. Saves valid TemporalConv features **before** the reference BiLSTM, together with video identity, labels, source-frame hash and producer information.
7. Reloads the first three saved tensors, runs the original remaining model, and checks both logits and decoded predictions match the original route. `--verify-first` can increase this count.

Output:

```text
dev_first5/
  metadata.json          # Checkpoint, preprocessing and source-code fingerprints
  index.json             # Finished list of videos, lengths and file checksums
  reference_report.json  # Greedy reference predictions, WER and equivalence checks
  tensors/
    <video-hash>.pt       # One video's float32 features and identity
```

An index is published only when the whole requested extraction finishes. Training rejects incomplete caches. The source dataset is read in place and is not rewritten.

**About scores:** the authors list 18.9% dev and 20.5% test WER. Those are published numbers, not results measured here. This implementation starts with greedy decoding and raw-gloss scoring. The upstream released evaluation uses beam decoding and its scoring procedure. Do not claim exact reproduction of the published numbers from this greedy report. Matching the official decoder/scoring and investigating differences remains a reference-validation task.

## 6. Extract the full train and dev feature collections

Use new output folders; a five-video cache cannot become a full cache by changing its manifest or limit during resume.

```powershell
python -m sign_language.cli.main extract-features --manifest data/prepared/phase1/manifests/reference/train.jsonl --output data/processed/corrnet_reference/train
python -m sign_language.cli.main extract-features --manifest data/prepared/phase1/manifests/reference/dev.jsonl --output data/processed/corrnet_reference/dev
```

This processes 7,096 training and 519 development videos and may take considerable time. It handles one video at a time. The final feature tensors are roughly 1 GB across all splits, plus file metadata; this is an estimate, not a measured cache size.

If extraction stops, rerun the **same command** with `--resume`. Finished tensors are reused, and the first verification videos are checked again. Keep the original frames unchanged during an extraction/resume cycle. Frame hashes are captured when videos are read; resume does not reread every previously completed video.

Checkpoint, source code, preprocessing, device type, and library-version changes require a new cache. An alternative is `--device cpu`, which uses system memory and is slower. Keep the device type consistent for train/dev caches; this implementation conservatively rejects mixed producers. Do not split a video into arbitrary temporal chunks to fix GPU memory issues, because CorrNet uses temporal context.

## 7. Understand and train your custom model (plan steps 2.5–2.8)

The model is:

```text
1024-dimensional cached features
  -> Linear 1024 to 512 + LayerNorm + ReLU + dropout 0.1
  -> two residual temporal Conv1d blocks, kernel size 3
  -> two-layer BiLSTM, 256 units in each direction
  -> head A: Linear 512 to vocabulary size including blank
```

The temporal encoder preserves the number of time steps. Residual blocks use LayerNorm per time step rather than batch statistics, and mask padding after each block. The BiLSTM uses packed sequences so added batch padding cannot change the valid predictions.

CTC teaches label order without requiring a timestamp for every sign. Targets cannot contain blank. Repeated target labels require extra output steps; impossible alignments fail clearly instead of quietly contributing zero loss.

First overfit a handful of **real training clips** once real features exist:

```powershell
python -m sign_language.cli.main overfit --cache data/processed/corrnet_reference/train --output runs/real_clip_overfit --limit 4
```

This trains on the first four cached training clips for at most 300 updates, stopping when it predicts all their sequences correctly. It writes `overfit_report.json` with references and predictions, and `overfit.pt` with the saved model. For this debugging check it uses constant learning rate and zero weight decay to make memorization easier. If it does not learn the clips, inspect the report, increase `--max-steps`, or use a separate config to investigate model size and learning rate before a costly full run. This score measures memorization of training clips; it is not dev or test accuracy. The overfit checkpoint is separate from baseline training and has no resume command.

A smaller debugging configuration is available with `--config configs/training/overfit_debug.yaml`. It uses fewer hidden units, no dropout and a higher learning rate, so the basic learning check is cheaper. Use the default `heads.yaml` configuration for the planned full baseline.

The full run command is:

```powershell
python -m sign_language.cli.main train --train-cache data/processed/corrnet_reference/train --dev-cache data/processed/corrnet_reference/dev --output runs/head_a_seed42
```

The command uses `configs/training/heads.yaml`. Its starting settings are 40 epochs, batch size 8, Adam at learning rate 0.001, weight decay 0.001, gradient clipping and cosine learning-rate decay. An **epoch** is one pass over the selected training videos. These are starting settings, not a promise of a particular score.

To train a chosen subset, add `--train-manifest path/to/small_train.jsonl --dev-manifest path/to/small_dev.jsonl`. Every selected record must match its cached record exactly. Use a separate copy of the YAML configuration with `--config path/to/my_config.yaml` for different model sizes or training settings.

To run only one epoch as a check while preserving the planned 40-epoch learning-rate schedule:

```powershell
python -m sign_language.cli.main train --train-cache data/processed/corrnet_reference/train --dev-cache data/processed/corrnet_reference/dev --output runs/head_a_seed42 --stop-after-epochs 1
```

Use either the full-run command or this first-epoch command for a fresh run, not both on an already-existing folder.

## 8. Resume and inspect a run

```powershell
python -m sign_language.cli.main train --train-cache data/processed/corrnet_reference/train --dev-cache data/processed/corrnet_reference/dev --output runs/head_a_seed42 --resume
```

Resume restores the model, optimizer, scheduler, random-number states and data-shuffle generator. It resumes at the last **completed epoch**, not halfway through a batch. Configuration, data, vocabulary, implementation and recorded environment must match. If they change, start a new run folder.

```text
runs/head_a_seed42/
  run.json                   # Settings, vocabulary and feature/data provenance
  history.json               # Loss, dev WER and learning rate for every epoch
  last.pt                    # Resume from the latest completed epoch
  best.pt                    # Model with the lowest dev WER
  best_dev_predictions.json  # Reference and predicted glosses for each dev video
```

Do not use the test set to choose `best.pt`. The training command enforces dev split selection and rejects train/dev video-ID overlap. CPU resume reproducibility is tested exactly; CUDA results may still vary across devices or library builds.

## 9. Evaluate after choosing the model

For an additional dev report:

```powershell
python -m sign_language.cli.main evaluate --checkpoint runs/head_a_seed42/best.pt --cache data/processed/corrnet_reference/dev --output runs/head_a_seed42/dev_evaluation.json
```

Once model choices are fixed, extract and evaluate the final test set:

```powershell
python -m sign_language.cli.main extract-features --manifest data/prepared/phase1/manifests/reference/test.jsonl --output data/processed/corrnet_reference/test
python -m sign_language.cli.main evaluate --checkpoint runs/head_a_seed42/best.pt --cache data/processed/corrnet_reference/test --output runs/head_a_seed42/test_evaluation.json
```

The JSON report gives substitutions (wrong signs), deletions (missing signs), insertions (extra signs), total reference labels, corpus WER and per-video predictions. WER is total errors divided by total reference labels, multiplied by 100. Lower is better; enough extra predictions can make it exceed 100%.

Greedy decoding merges adjacent repeats before removing blanks: `A A blank A` becomes `A A`. Beam search is a possible later alternative because it considers several candidate paths; it is not implemented in this Phase 2 command.

## 10. Know which files to edit

| File under `src/sign_language/` | What it does |
|---|---|
| `data/transforms.py` | Turns the original PNG images into reference-model inputs |
| `models/backbones/corrnet.py` | Loads the released architecture and matching checkpoint |
| `features/extractor.py`, `features/cache.py` | Extracts, saves, resumes and validates cached features |
| `features/validation.py` | Checks saved features preserve downstream outputs |
| `data/dataset.py`, `data/collate.py` | Reads feature files and forms padded batches |
| `models/temporal/encoder.py` | Reads temporal patterns using convolutions and a BiLSTM |
| `models/heads/linear.py`, `models/recognizer.py` | Produces head-A label scores |
| `losses/ctc.py` | Computes the CTC learning loss and validates alignments |
| `decoding/greedy.py` | Converts time-step scores into a label sequence |
| `evaluation/wer.py` | Counts prediction mistakes |
| `training/trainer.py`, `training/loops.py` | Runs training and dev evaluation |
| `training/checkpoints.py` | Saves and reloads training state |
| `cli/main.py` | Defines the commands above |

The CLI reads `configs/training/heads.yaml` by default. Phase 3 adds active `configs/experiments/head_*.yaml` configs selected through `--config`; see the [Phase 3 guide](phase3.md). The API, deployment and clean-backbone training remain later work.

## 11. Run the automated tests

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_phase*.py' -v
```

The tests cover Phase 1, CTC, repeated signs, padding, WER, cache identity and corruption, extraction/resume orchestration, model backward computation, dev/test separation, and exact CPU training resume. They use temporary fixtures, not your 40 GB dataset.

The pretrained feature cache is a known-sign reference. Because its backbone was trained on the full reference training set, using it with a reduced manifest does **not** create a clean unseen-sign experiment. Follow [the experiment rules](experiment_protocol.md) before that later phase.
