# Phase 1: prepare your dataset

Phase 1 has a working standard-library-only data preparation command. Phase 2 model and training commands are explained in [the Phase 2 guide](phase2.md); the API and deployment remain placeholders. Python 3.10 or newer is required. No deep-learning dependencies are required to prepare data.

## Run from the project folder in PowerShell

```powershell
$env:PYTHONPATH = 'src'
python -m sign_language.cli.main prepare-data
```

The default input is your existing PHOENIX-2014-T folder. The default output is `data/prepared/phase1`. This output is a complete preparation bundle containing reference and reduced manifests, dictionaries, the held-out policy, validation report, and an empty experiment register. Original images and annotations are never rewritten.

The command refuses to overwrite an existing output. For a new preparation version:

```powershell
python -m sign_language.cli.main prepare-data --output data/prepared/phase1-v2 --seed 42
```

To install a console entry point later, use a dedicated environment and `python -m pip install -e .`. The source-based command above does not require installation. The empty YAML scaffold files are not read by this command; its options are explicit CLI arguments. Use `--help` to inspect them.

## Validation performed

Every clip folder is resolved using either flat images or the nested `1/` layout. Ambiguous layouts fail. Every PNG filename and nonzero file size is checked; indices must start at one and have no gaps. By default the first PNG header of every clip is checked for 210x260 dimensions. `--all-png-headers` checks every header. Neither option fully decodes image payloads; this is not a guarantee against every corrupt image.

Annotation columns, IDs, labels, signers, and duplicate IDs across splits are checked. Paths in generated manifests are relative to the dataset root. Translation text is retained separately and is never substituted for gloss targets.

## Held-out experiment

The initial split randomly samples 10% of ordinary training gloss types with seed 42. It is a reproducible pilot, not a linguistically reviewed sign-family split. All videos containing selected glosses are excluded. Glosses that incidentally disappear are listed separately. Preparation fails if less than 20% of training videos remain.

For a reviewed selection, supply `--held-out-file path/to/glosses.txt`, with one exact training gloss per line. Freeze your chosen split before inspecting recognition results. Seen-only development manifests exclude any label absent from reduced training. Held-out evaluation manifests contain at least one deliberately held-out gloss; full official manifests are retained for full-sequence evaluation. Dev/test labels never determine random training-label selection.

Dictionaries reserve ID zero for `<blank>` and sort ordinary target labels. These are project-owned IDs, NOT verified CorrNet checkpoint IDs. Phase 2 must import and validate the checkpoint's dictionary before reference inference.

## Run checks

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p test_phase1.py -v
```

These Phase 1 tests use temporary small fixtures, not your dataset. No model is trained by them. See [the Phase 2 status](../reports/phase2_summary.md) for the separate GPU, image-preprocessing and model checks.
