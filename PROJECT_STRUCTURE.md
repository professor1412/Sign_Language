> Update: Phase 1 data preparation is now implemented. See [README.md](README.md) and [setup instructions](docs/setup.md). The scaffold snapshot below describes the original empty structure, not current implementation status.

# Project file and folder guide

This is a production-oriented layout for the five-phase sign-language research project. It provides places for code, tests, configuration, results, and an optional prediction service. **It is a scaffold, not a working or production-ready application.**

## What has been created

- Empty Python source files and package markers.
- Empty configuration, test, and deployment files for you to fill in.
- Documentation explaining where work belongs.
- Active `.gitignore` and `.dockerignore` rules to keep the dataset and generated assets out of normal commits and container build contexts.
- Empty `.gitkeep` files to preserve storage folders. These files have no runtime behavior.

No application code, dependency installation, training, model downloads, API service, Git initialization, or deployment was performed. The original plan and existing dataset were preserved. Workflow templates end in `.example` so an empty CI configuration does not run automatically.

## Where should I write things?

| If you are working on... | Use this location |
|---|---|
| Reading videos, labels, and experiment splits | `src/sign_language/data/` |
| Saving reusable numerical video features | `src/sign_language/features/` |
| Word cleaning and text embeddings | `src/sign_language/text/` |
| Video reader, sequence reader, and answer choosers | `src/sign_language/models/` |
| CTC and optional extra learning signals | `src/sign_language/losses/` |
| Turning scores into gloss sequences | `src/sign_language/decoding/` |
| Training and saving/resuming progress | `src/sign_language/training/` |
| WER, unseen precision/recall, and error analysis | `src/sign_language/evaluation/` |
| Loading a model and predicting new input | `src/sign_language/inference/` |
| Terminal commands that call the above components | `src/sign_language/cli/` |
| A future prediction API | `src/sign_language/service/` |
| Experiment settings rather than program logic | `configs/` |
| Checks that your implementation is correct | `tests/` |
| Generated data lists and label mappings | `data/` |
| Large model files and learned feature caches | `artifacts/` |
| Individual experiment outputs and logs | `runs/` |
| Final reviewed tables and write-ups | `reports/` |
| Setup, architecture, and operating instructions | `docs/` |
| Container and monitoring setup, after inference works | `deployment/` |

## Complete scaffold tree

The dataset is abbreviated to its confirmed top-level structure; thousands of frame files are deliberately not listed. Every other entry below is part of this scaffold or the existing plan.

```text
Sign_Language/
|-- .github/
|   `-- workflows/
|       |-- checks.yml.example
|       `-- release.yml.example
|-- artifacts/
|   |-- checkpoints/
|   |   |-- heads/
|   |   |   `-- .gitkeep
|   |   |-- reduced/
|   |   |   `-- .gitkeep
|   |   `-- reference/
|   |       `-- .gitkeep
|   |-- exports/
|   |   `-- .gitkeep
|   |-- features/
|   |   |-- reduced/
|   |   |   `-- .gitkeep
|   |   `-- reference/
|   |       `-- .gitkeep
|   `-- pretrained/
|       |-- corrnet/
|       |   `-- .gitkeep
|       |-- generic/
|       |   `-- .gitkeep
|       `-- text/
|           `-- .gitkeep
|-- configs/
|   |-- data/
|   |   |-- held_out_split.yaml
|   |   `-- phoenix2014t.yaml
|   |-- evaluation/
|   |   |-- known_signs.yaml
|   |   `-- unseen_signs.yaml
|   |-- experiments/
|   |   |-- clean_backbone.yaml
|   |   |-- head_a.yaml
|   |   |-- head_a_norm.yaml
|   |   |-- head_b.yaml
|   |   |-- head_b_random.yaml
|   |   |-- head_b_shuffled.yaml
|   |   |-- head_c.yaml
|   |   |-- reference.yaml
|   |   |-- spiking.yaml
|   |   |-- superposition.yaml
|   |   `-- unseen_signs.yaml
|   |-- inference/
|   |   |-- local.yaml
|   |   `-- service.yaml
|   |-- models/
|   |   |-- corrnet.yaml
|   |   |-- heads.yaml
|   |   |-- s3d.yaml
|   |   |-- spiking.yaml
|   |   `-- temporal.yaml
|   |-- text/
|   |   |-- alternatives.yaml
|   |   `-- fasttext.yaml
|   |-- training/
|   |   |-- backbone.yaml
|   |   `-- heads.yaml
|   |-- base.yaml
|   |-- logging.yaml
|   `-- paths.yaml
|-- data/
|   |-- manifests/
|   |   |-- reduced/
|   |   |   `-- .gitkeep
|   |   `-- reference/
|   |       `-- .gitkeep
|   |-- previews/
|   |   `-- .gitkeep
|   |-- processed/
|   |   `-- .gitkeep
|   |-- splits/
|   |   `-- .gitkeep
|   |-- text/
|   |   |-- descriptions/
|   |   |   `-- .gitkeep
|   |   `-- embeddings/
|   |       |-- alternatives/
|   |       |   `-- .gitkeep
|   |       `-- fasttext/
|   |           `-- .gitkeep
|   |-- vocabularies/
|   |   |-- reduced/
|   |   |   `-- .gitkeep
|   |   `-- reference/
|   |       `-- .gitkeep
|   `-- README.md
|-- deployment/
|   |-- docker/
|   |   |-- compose.yaml.example
|   |   `-- Dockerfile
|   `-- monitoring/
|       |-- alerts.yaml.example
|       `-- prometheus.yaml.example
|-- docs/
|   |-- architecture.md
|   |-- data_contracts.md
|   |-- deployment.md
|   |-- evaluation.md
|   |-- experiment_protocol.md
|   |-- inference.md
|   |-- model_card.md
|   |-- operations.md
|   |-- setup.md
|   `-- troubleshooting.md
|-- external/
|   |-- corrnet/
|   |   `-- .gitkeep
|   |-- s3d/
|   |   `-- .gitkeep
|   `-- spiking/
|       `-- .gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- phoenix-2014-T.v3/
|   `-- PHOENIX-2014-T-release-v3/
|       |-- PHOENIX-2014-T/
|       |   |-- annotations/
|       |   |   `-- manual/
|       |   |       `-- [existing annotation files]
|       |   |-- evaluation/
|       |   |   `-- [existing evaluation files]
|       |   |-- features/
|       |   |   `-- fullFrame-210x260px/
|       |   |       |-- dev/
|       |   |       |   `-- [existing frame sequences]
|       |   |       |-- test/
|       |   |       |   `-- [existing frame sequences]
|       |   |       `-- train/
|       |   |           `-- [existing frame sequences]
|       |   `-- models/
|       |       `-- [existing language-model files]
|       `-- README
|-- reports/
|   |-- error_analysis/
|   |   `-- .gitkeep
|   |-- figures/
|   |   `-- .gitkeep
|   |-- final/
|   |   `-- .gitkeep
|   `-- tables/
|       `-- .gitkeep
|-- runs/
|   |-- extensions/
|   |   `-- .gitkeep
|   |-- known_signs/
|   |   `-- .gitkeep
|   |-- reference/
|   |   `-- .gitkeep
|   `-- unseen_signs/
|       `-- .gitkeep
|-- src/
|   `-- sign_language/
|       |-- cli/
|       |   |-- commands/
|       |   |   |-- __init__.py
|       |   |   |-- build_embeddings.py
|       |   |   |-- evaluate.py
|       |   |   |-- export.py
|       |   |   |-- extract_features.py
|       |   |   |-- predict.py
|       |   |   |-- prepare_data.py
|       |   |   `-- train.py
|       |   |-- __init__.py
|       |   `-- main.py
|       |-- core/
|       |   |-- __init__.py
|       |   |-- exceptions.py
|       |   |-- logging.py
|       |   |-- provenance.py
|       |   |-- reproducibility.py
|       |   `-- settings.py
|       |-- data/
|       |   |-- __init__.py
|       |   |-- annotations.py
|       |   |-- collate.py
|       |   |-- dataset.py
|       |   |-- frame_reader.py
|       |   |-- manifest.py
|       |   |-- samplers.py
|       |   |-- splits.py
|       |   |-- transforms.py
|       |   |-- validation.py
|       |   `-- vocabulary.py
|       |-- decoding/
|       |   |-- __init__.py
|       |   |-- beam.py
|       |   `-- greedy.py
|       |-- evaluation/
|       |   |-- __init__.py
|       |   |-- alignment.py
|       |   |-- analysis.py
|       |   |-- reporting.py
|       |   |-- unseen.py
|       |   `-- wer.py
|       |-- features/
|       |   |-- __init__.py
|       |   |-- cache.py
|       |   |-- extractor.py
|       |   `-- validation.py
|       |-- inference/
|       |   |-- __init__.py
|       |   |-- export.py
|       |   |-- predictor.py
|       |   |-- validation.py
|       |   `-- vocabulary_expansion.py
|       |-- losses/
|       |   |-- __init__.py
|       |   |-- auxiliary.py
|       |   `-- ctc.py
|       |-- models/
|       |   |-- backbones/
|       |   |   |-- __init__.py
|       |   |   |-- base.py
|       |   |   |-- corrnet.py
|       |   |   |-- s3d.py
|       |   |   `-- spiking.py
|       |   |-- heads/
|       |   |   |-- __init__.py
|       |   |   |-- controls.py
|       |   |   |-- linear.py
|       |   |   |-- normalized_linear.py
|       |   |   |-- projected_text.py
|       |   |   `-- text_similarity.py
|       |   |-- temporal/
|       |   |   |-- __init__.py
|       |   |   `-- encoder.py
|       |   |-- __init__.py
|       |   |-- factory.py
|       |   `-- recognizer.py
|       |-- service/
|       |   |-- routes/
|       |   |   |-- __init__.py
|       |   |   |-- health.py
|       |   |   |-- model_info.py
|       |   |   `-- predictions.py
|       |   |-- __init__.py
|       |   |-- app.py
|       |   |-- dependencies.py
|       |   |-- middleware.py
|       |   |-- observability.py
|       |   `-- schemas.py
|       |-- text/
|       |   |-- __init__.py
|       |   |-- descriptions.py
|       |   |-- embedding_table.py
|       |   |-- fasttext_encoder.py
|       |   |-- normalization.py
|       |   `-- sentence_encoder.py
|       |-- training/
|       |   |-- __init__.py
|       |   |-- callbacks.py
|       |   |-- checkpoints.py
|       |   |-- loops.py
|       |   |-- optimizers.py
|       |   `-- trainer.py
|       `-- __init__.py
|-- tests/
|   |-- fixtures/
|   |   |-- annotations/
|   |   |   `-- .gitkeep
|   |   |-- features/
|   |   |   `-- .gitkeep
|   |   `-- frames/
|   |       `-- .gitkeep
|   |-- integration/
|   |   |-- test_data_pipeline.py
|   |   |-- test_feature_equivalence.py
|   |   |-- test_service.py
|   |   |-- test_training_resume.py
|   |   `-- test_vocabulary_expansion.py
|   |-- smoke/
|   |   `-- test_prediction_bundle.py
|   |-- unit/
|   |   |-- test_annotations.py
|   |   |-- test_cache.py
|   |   |-- test_collate.py
|   |   |-- test_ctc.py
|   |   |-- test_decoding.py
|   |   |-- test_heads.py
|   |   |-- test_input_validation.py
|   |   |-- test_metrics.py
|   |   |-- test_splits.py
|   |   `-- test_vocabulary.py
|   `-- conftest.py
|-- var/
|   |-- logs/
|   |   `-- .gitkeep
|   |-- tmp/
|   |   `-- .gitkeep
|   `-- uploads/
|       `-- .gitkeep
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- .pre-commit-config.yaml
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- PROJECT_STRUCTURE.md
|-- pyproject.toml
|-- README.md
|-- SECURITY.md
`-- SIGN_LANGUAGE_PROJECT_PLAN.md
```

## What to put in each implementation file

The descriptions below are instructions for your future implementation. They are not code already present in the files. All Python files, YAML/TOML placeholders, test files, and Dockerfile are empty.

### `project root`

| File | What you will write here |
|---|---|
| `.env.example` | List configuration variable names and safe example values; never put real secrets here. |
| `.pre-commit-config.yaml` | Configure checks to run before commits once the checks are implemented. |
| `CHANGELOG.md` | Record versioned changes after implementation starts. |
| `CONTRIBUTING.md` | Explain installation, coding conventions, reviews, and contribution steps. |
| `SECURITY.md` | Document reporting of security problems and handling of service uploads. |
| `pyproject.toml` | Declare the Python package, supported Python versions, dependencies, command-line entry point, and lint/test settings. |

### `.github/workflows`

| File | What you will write here |
|---|---|
| `checks.yml.example` | Template location for future lint, type, and test CI. Rename to .yml only after implementing it. |
| `release.yml.example` | Template location for future release builds and artifact checks. Rename only when ready. |

### `configs`

| File | What you will write here |
|---|---|
| `base.yaml` | Common settings shared by experiments. |
| `logging.yaml` | Log format, destination, and verbosity. |
| `paths.yaml` | External dataset path and generated-output locations; the actual dataset root is documented in data/README.md. |

### `configs/data`

| File | What you will write here |
|---|---|
| `held_out_split.yaml` | Held-out vocabulary rules, seed, development policy, and minimum remaining-data checks. |
| `phoenix2014t.yaml` | Frame source, annotation columns, preprocessing, and loader settings for PHOENIX-2014-T. |

### `configs/evaluation`

| File | What you will write here |
|---|---|
| `known_signs.yaml` | Known-label WER, decoder, and comparison settings. |
| `unseen_signs.yaml` | Expanded candidate vocabulary, unseen precision/recall, and subgroup evaluation settings. |

### `configs/experiments`

| File | What you will write here |
|---|---|
| `clean_backbone.yaml` | Phase 4: backbone trained only on the permitted reduced split. |
| `head_a.yaml` | Phase 2/3: standard classifier experiment. |
| `head_a_norm.yaml` | Phase 3: normalized-classifier control. |
| `head_b.yaml` | Phase 3: fixed text-embedding head. |
| `head_b_random.yaml` | Phase 3: random-prototype control. |
| `head_b_shuffled.yaml` | Phase 3: shuffled-text control. |
| `head_c.yaml` | Phase 3: shared text-projection head. |
| `reference.yaml` | Phase 2: reference reproduction configuration. |
| `spiking.yaml` | Phase 5: optional spiking comparison. |
| `superposition.yaml` | Phase 5: optional whole-video summary analysis. |
| `unseen_signs.yaml` | Phase 4: fresh clean heads and append-at-inference evaluation. |

### `configs/inference`

| File | What you will write here |
|---|---|
| `local.yaml` | Local prediction and exported-model settings. |
| `service.yaml` | Optional API model selection, request limits, timeouts, and worker settings. |

### `configs/models`

| File | What you will write here |
|---|---|
| `corrnet.yaml` | CorrNet architecture and initialization settings. |
| `heads.yaml` | A, A-norm, B, C, B-rand, and B-shuf head choices and dimensions. |
| `s3d.yaml` | Optional alternative backbone settings. |
| `spiking.yaml` | Optional spiking-backbone extension settings. |
| `temporal.yaml` | The sequence reader shared by output-head comparisons. |

### `configs/text`

| File | What you will write here |
|---|---|
| `alternatives.yaml` | Optional sentence encoders or reviewed sign-description settings. |
| `fasttext.yaml` | Text model location, normalization policy, and embedding-table settings. |

### `configs/training`

| File | What you will write here |
|---|---|
| `backbone.yaml` | Clean reduced-split backbone training settings. |
| `heads.yaml` | Cached-feature training settings shared fairly between head variants. |

### `deployment/docker`

| File | What you will write here |
|---|---|
| `Dockerfile` | Future inference container definition; package inference artifacts rather than training data. |
| `compose.yaml.example` | Future local service deployment template; inactive until completed. |

### `deployment/monitoring`

| File | What you will write here |
|---|---|
| `alerts.yaml.example` | Optional future alert rules for failed predictions, latency, and resource exhaustion. |
| `prometheus.yaml.example` | Optional future service-metrics collection configuration. |

### `docs`

| File | What you will write here |
|---|---|
| `architecture.md` | Explain component relationships, interfaces, and dependency boundaries. |
| `data_contracts.md` | Define manifest, batch, cache, vocabulary, and annotation formats. |
| `deployment.md` | Document packaging, service configuration, rollout, rollback, and resource limits. |
| `evaluation.md` | Document decoding, sequence alignment, metrics, and reproducible evaluation. |
| `experiment_protocol.md` | Declare seen/unseen splits, model selection, allowed exposure, and controls. |
| `inference.md` | Explain local prediction, optional API requests, and vocabulary expansion. |
| `model_card.md` | Describe a trained model, data, measured results, intended use, and limits. |
| `operations.md` | Document monitoring, retention/cleanup, incident handling, and model replacement. |
| `setup.md` | Document tested environment installation and GPU/CPU checks. |
| `troubleshooting.md` | Record common setup, data, CTC, cache, and memory failures and verified fixes. |

### `src/sign_language/cli`

| File | What you will write here |
|---|---|
| `main.py` | One command-line entry point; dispatch to commands without duplicating model logic. |

### `src/sign_language/cli/commands`

| File | What you will write here |
|---|---|
| `build_embeddings.py` | Create and audit text embedding tables. |
| `evaluate.py` | Evaluate a saved model under a declared protocol. |
| `export.py` | Create a documented inference bundle. |
| `extract_features.py` | Run feature extraction and cache validation. |
| `predict.py` | Predict one local video or frame sequence. |
| `prepare_data.py` | Run annotation checks, manifest generation, dictionary creation, and split preparation. |
| `train.py` | Start or resume a configured backbone/head experiment. |

### `src/sign_language/core`

| File | What you will write here |
|---|---|
| `exceptions.py` | Define useful project-specific errors. |
| `logging.py` | Configure consistent logs without mixing log setup into model code. |
| `provenance.py` | Record code, data split, checkpoint, vocabulary, and preprocessing identities. |
| `reproducibility.py` | Manage seeds and reproducibility settings. |
| `settings.py` | Read and validate configuration, environment overrides, and required paths. |

### `src/sign_language/data`

| File | What you will write here |
|---|---|
| `annotations.py` | Read original PHOENIX annotation files while keeping gloss and translation fields distinct. |
| `collate.py` | Assemble variable-length examples into batches with padding and correct lengths. |
| `dataset.py` | Expose video records or cached features to the training framework. |
| `frame_reader.py` | Read ordered image sequences or supported video input without altering the original dataset. |
| `manifest.py` | Build and read per-video records: ID, source, split, labels, and lengths. |
| `samplers.py` | Group or order examples efficiently while preserving experiment rules. |
| `splits.py` | Choose held-out labels, exclude entire videos, and audit remaining seen-label coverage. |
| `transforms.py` | Implement resizing, crops, normalization, and controlled training augmentation. |
| `validation.py` | Validate data records, missing frames, duplicates, and training/evaluation separation. |
| `vocabulary.py` | Build dictionaries and preserve blank, special-token, and gloss-ID mappings. |

### `src/sign_language/decoding`

| File | What you will write here |
|---|---|
| `beam.py` | Wrap a selected beam decoder with explicit vocabulary and blank handling. |
| `greedy.py` | Apply greedy selection and the correct repeat-then-blank CTC collapse. |

### `src/sign_language/evaluation`

| File | What you will write here |
|---|---|
| `alignment.py` | Align full predicted and reference sequences using a fixed tie policy. |
| `analysis.py` | Analyze rare labels, related groups, confusion pairs, and representative errors. |
| `reporting.py` | Write machine-readable results and reproducible comparison tables. |
| `unseen.py` | Calculate unseen occurrence/type recall, precision, and seen-only/mixed sentence metrics. |
| `wer.py` | Calculate sequence substitutions, deletions, insertions, and WER. |

### `src/sign_language/features`

| File | What you will write here |
|---|---|
| `cache.py` | Read and write cached features with source identities and schema versions. |
| `extractor.py` | Extract the chosen frozen-backbone features and valid output lengths. |
| `validation.py` | Check cache shape, lengths, provenance, and agreement with original predictions. |

### `src/sign_language/inference`

| File | What you will write here |
|---|---|
| `export.py` | Bundle model weights, preprocessing, vocabulary, and compatibility metadata. |
| `predictor.py` | Load a selected model and return gloss sequences through one shared prediction interface. |
| `validation.py` | Validate input size, frame limits, formats, and model/vocabulary compatibility. |
| `vocabulary_expansion.py` | Append new text-defined labels without weight updates and preserve ID mappings. |

### `src/sign_language/losses`

| File | What you will write here |
|---|---|
| `auxiliary.py` | Optional segment/text or bag-of-glosses losses after the core experiment works. |
| `ctc.py` | Prepare CTC log probabilities and check targets, blank IDs, and valid lengths. |

### `src/sign_language/models`

| File | What you will write here |
|---|---|
| `factory.py` | Build model components from an explicit validated configuration. |
| `recognizer.py` | Connect the video/feature input, sequence reader, and chosen output head. |

### `src/sign_language/models/backbones`

| File | What you will write here |
|---|---|
| `base.py` | Define a common video-feature interface and output-length contract. |
| `corrnet.py` | Adapt reviewed external CorrNet code and checkpoint loading to the project interface. |
| `s3d.py` | Optional S3D backbone adapter. |
| `spiking.py` | Optional spiking-backbone adapter; verify the actual implementation before integrating. |

### `src/sign_language/models/heads`

| File | What you will write here |
|---|---|
| `controls.py` | Fixed random and shuffled prototypes with saved assignments for B-rand and B-shuf. |
| `linear.py` | Head A: learned standard classifier. |
| `normalized_linear.py` | A-norm: normalized learned classifier control. |
| `projected_text.py` | Head C: shared learned text projection, normalized comparison, and temperature. |
| `text_similarity.py` | Head B: cosine scores against fixed text vectors plus learned blank/special rows. |

### `src/sign_language/models/temporal`

| File | What you will write here |
|---|---|
| `encoder.py` | Implement the shared sequence reader and correct padding behavior. |

### `src/sign_language/service`

| File | What you will write here |
|---|---|
| `app.py` | Optional prediction API assembly; keep training outside the request process. |
| `dependencies.py` | Manage model startup/loading and per-request dependencies. |
| `middleware.py` | Implement request IDs, error mapping, authentication hooks, and request handling policies. |
| `observability.py` | Expose latency, failures, throughput, and resource metrics without logging raw user videos. |
| `schemas.py` | Define request, response, and error formats. |

### `src/sign_language/service/routes`

| File | What you will write here |
|---|---|
| `health.py` | Optional API health/readiness checks including whether the model is loaded. |
| `model_info.py` | Optional endpoint describing the loaded version, supported vocabulary, and limits. |
| `predictions.py` | Optional prediction endpoints using inference/predictor.py. |

### `src/sign_language/text`

| File | What you will write here |
|---|---|
| `descriptions.py` | Load reviewed sign descriptions and record their independent sources. |
| `embedding_table.py` | Assemble vectors in exact ID order and handle frozen versus learned rows. |
| `fasttext_encoder.py` | Load German fastText and obtain ordinary-gloss embeddings. |
| `normalization.py` | Create text inputs while preserving original target labels; report collisions. |
| `sentence_encoder.py` | Optional adapter for alternative text encoders with documented extraction rules. |

### `src/sign_language/training`

| File | What you will write here |
|---|---|
| `callbacks.py` | Handle logging, early stopping, and best-checkpoint selection from permitted dev metrics. |
| `checkpoints.py` | Save and restore model, optimizer, scheduler, random state, and run metadata. |
| `loops.py` | Implement one training epoch and one validation pass. |
| `optimizers.py` | Build optimizer parameter groups and learning-rate schedules. |
| `trainer.py` | Coordinate training, validation, checkpoints, and interruption/resume behavior. |

### `tests`

| File | What you will write here |
|---|---|
| `conftest.py` | Shared test fixtures; use tiny synthetic or permitted examples, not the entire 40 GB dataset. |

### `tests/integration`

| File | What you will write here |
|---|---|
| `test_data_pipeline.py` | Tiny annotation-to-batch flow with known expected IDs and lengths. |
| `test_feature_equivalence.py` | Original predictions versus cached-feature predictions. |
| `test_service.py` | Optional API startup, health, prediction errors, and response contracts. |
| `test_training_resume.py` | Save/reload/resume behavior and recovery of required training state. |
| `test_vocabulary_expansion.py` | New candidates can be appended without changing trained weights. |

### `tests/smoke`

| File | What you will write here |
|---|---|
| `test_prediction_bundle.py` | A packaged model loads and predicts a tiny supported example. |

### `tests/unit`

| File | What you will write here |
|---|---|
| `test_annotations.py` | Annotation parsing and separation of gloss versus translation fields. |
| `test_cache.py` | Cache metadata, length checks, and rejection of incompatible sources. |
| `test_collate.py` | Padding, ordering, and true lengths for variable-length examples. |
| `test_ctc.py` | Repeated targets, impossible lengths, blank handling, and finite loss checks. |
| `test_decoding.py` | Correct repeat collapse and blank removal, including repeated signs. |
| `test_heads.py` | Head dimensions, frozen embeddings, learned special rows, and temperature behavior. |
| `test_input_validation.py` | Prediction input limits and invalid-format handling. |
| `test_metrics.py` | WER alignment, unseen precision/recall, and empty-denominator cases. |
| `test_splits.py` | Whole-video exclusion, disjoint splits, and incidentally unseen labels. |
| `test_vocabulary.py` | Stable IDs, blank/special handling, and normalization collisions. |

## Storage folders: outputs go here, implementation does not

| Folder | Purpose |
|---|---|
| `data/manifests/reference/` | Generated manifests for the official/full-data control experiments. |
| `data/manifests/reduced/` | Generated allowed-video manifests for clean held-out-sign experiments. |
| `data/splits/` | Versioned held-out lists, seeds, split policies, and coverage reports. |
| `data/vocabularies/reference/` | Checkpoint-compatible full-data label maps. |
| `data/vocabularies/reduced/` | Seen, held-out, and expanded candidate maps plus ID conversions. |
| `data/text/descriptions/` | Reviewed descriptions and independent source records. |
| `data/text/embeddings/fasttext/` | Generated fastText tables and normalization/collision records. |
| `data/text/embeddings/alternatives/` | Optional alternative text tables with model provenance. |
| `data/processed/` | Generated preprocessing outputs if needed; never another copy of the whole raw dataset. |
| `data/previews/` | Small local inspection samples, excluded from version control. |
| `artifacts/pretrained/corrnet/` | Downloaded reference checkpoint and provenance. |
| `artifacts/pretrained/generic/` | Generic visual initialization weights and provenance. |
| `artifacts/pretrained/text/` | Downloaded text encoder weights, including the large fastText binary. |
| `artifacts/features/reference/` | Feature cache from the full-data reference backbone; unsuitable for clean zero-shot claims. |
| `artifacts/features/reduced/` | Feature cache from the clean reduced-split backbone. |
| `artifacts/checkpoints/reference/` | Reference-related model snapshots. |
| `artifacts/checkpoints/reduced/` | Clean reduced-split backbone snapshots. |
| `artifacts/checkpoints/heads/` | Head snapshots stored in future run-specific subfolders with provenance. |
| `artifacts/exports/` | Versioned inference bundles produced later. |
| `runs/reference/` | Future run folders for reference reproduction. |
| `runs/known_signs/` | Future run folders for A/B/C and control comparisons. |
| `runs/unseen_signs/` | Future run folders for clean unseen-sign comparisons. |
| `runs/extensions/` | Future run folders for optional spiking and summary analyses. |
| `reports/tables/` | Final reviewed comparison tables. |
| `reports/figures/` | Final plots and error-analysis figures. |
| `reports/error_analysis/` | Reviewed example predictions and interpretations. |
| `reports/final/` | Final research write-up and supporting export files. |
| `notebooks/` | Optional exploratory notebooks; reusable implementation belongs in src. |
| `external/corrnet/` | Location reserved for reviewed upstream code if later vendored; nothing downloaded now. |
| `external/s3d/` | Optional reviewed upstream S3D source, if later needed. |
| `external/spiking/` | Optional reviewed upstream spiking source, if later needed. |
| `tests/fixtures/annotations/` | Tiny synthetic/permitted annotation examples for tests. |
| `tests/fixtures/frames/` | Tiny synthetic/permitted frame examples for tests. |
| `tests/fixtures/features/` | Tiny synthetic feature arrays for tests. |
| `var/logs/` | Local runtime service logs; not research source files. |
| `var/uploads/` | Temporary optional service uploads requiring implementation of limits and cleanup. |
| `var/tmp/` | Disposable local runtime temporary files. |

## Which files should you fill in first?

Follow the same five phases as the plan. You do not need to implement every placeholder immediately.

| Phase | Begin with | Complete result |
|---|---|---|
| 1: setup and data | `pyproject.toml`, `configs/paths.yaml`, `core/settings.py`, `data/annotations.py`, `data/manifest.py`, `data/vocabulary.py`, `data/splits.py`, `data/validation.py` | Checked dataset records, vocabulary, and experiment splits. |
| 2: first recognition system | `models/backbones/corrnet.py`, `features/`, `models/temporal/encoder.py`, `models/heads/linear.py`, `losses/ctc.py`, `decoding/greedy.py`, `training/` | Verified reference, cache, and working A baseline. |
| 3: text-based comparison | `text/`, the other files in `models/heads/`, `configs/experiments/head_*.yaml`, `evaluation/` | Fair A/B/C and control comparisons. |
| 4: unseen-sign experiment | `core/provenance.py`, clean split/configuration files, `inference/vocabulary_expansion.py`, `evaluation/unseen.py` | Clean training history and measured held-out-sign performance. |
| 5: finish the project | `evaluation/reporting.py`, `inference/export.py`, `reports/`, `docs/`, relevant tests | Reproducible results and a documented inference bundle. |

Source paths in this table are relative to `src/sign_language/` unless they explicitly start with `configs/` or name a root file/folder. Implement the CLI commands as thin entry points calling the corresponding source modules.

The API, containers, monitoring, S3D, spiking model, alternate text encoders, and extra losses are later or optional work. Do not let their empty placeholders distract you from the first working data pipeline.

## Boundaries that keep the project maintainable

1. Put reusable logic in `src/sign_language/`, not in notebooks or configuration files.
2. Keep paths and experiment choices in configuration; do not hard-code one laptop's absolute paths into model code.
3. Let CLI commands and API routes call shared modules. Avoid implementing separate prediction logic for each interface.
4. Keep training outside the API request process. The service should load a selected, tested inference bundle.
5. Distinguish original frame sequences from learned feature caches. The dataset's `features/` directory is not your model cache.
6. Keep `reference` and `reduced` assets separate and attach provenance to every run. Do not reuse an exposed backbone/cache for a clean unseen-sign claim.
7. Keep original annotation files untouched. Use manifests to select examples rather than copying or deleting source videos.
8. Tests should use tiny synthetic or permitted fixtures. Full-dataset checks are explicit experiments, not a requirement for every unit test run.
9. Downloaded third-party code belongs under a deliberate external-dependency policy. Record upstream revision and license before using it; no upstream code was fetched in this scaffold.

## What goes inside a future run folder?

Create a unique folder for each actual experiment, such as `runs/known_signs/<run-id>/`. The following are planned outputs, **not files created now**:

| Future output | What it records |
|---|---|
| `resolved_config.yaml` | Exact settings after all overrides. |
| `provenance.json` | Code revision, seed, dataset/split identity, vocabulary, and source checkpoints. |
| `metrics.jsonl` | Measurements over training or evaluation. |
| `predictions.jsonl` | Video IDs, predicted labels, references where allowed, and relevant output metadata. |
| `checkpoint_index.json` | Links or paths to the checkpoints stored for this run. |
| `logs/` | Diagnostic logs for that run. |
| `figures/` | Run-specific plots. |

Do not create fake JSON, fake checkpoints, or empty dependency lockfiles to imply measured results. Populate outputs only when real code generates them. Final reviewed evidence goes into `reports/`.

## What must be implemented before a production release?

The folders reserve places for these tasks; none is implemented yet:

- Input validation and limits for frames, duration, formats, uploads, and resource use.
- Reliable model loading, compatibility checks, clear errors, and health/readiness reporting.
- Tested label IDs, CTC decoding, padding, metrics, and vocabulary expansion.
- Saved configuration and provenance, correct training resume, and reproducible inference exports.
- Service access controls appropriate to its deployment, request timeouts, and temporary-file cleanup.
- Logs, operational metrics, dependency management, CI checks, and documented rollout/rollback.
- Evaluation on the intended data, a model card, and explicit recognition limitations.

Finish the core research system first. A directory layout can support these requirements, but only implementation, testing, and operating evidence can establish readiness.

## Existing dataset and source files

The confirmed dataset root is:

```text
phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
```

See [data/README.md](data/README.md) for its observed annotation and frame paths. The existing [beginner project plan](SIGN_LANGUAGE_PROJECT_PLAN.md) is unchanged. The large dataset has not been moved, copied, rewritten, or fully integrity-checked.
