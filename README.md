# Sign Language Recognition

Phase 1 data preparation, Phase 2 recognition/training, Phase 3 text heads, and Phase 4 clean unseen-sign study code are implemented. Start with the [Phase 2 guide](docs/phase2.md) for feature extraction, [Phase 3 guide](docs/phase3.md) for text heads, and [beginner Phase 4 guide](docs/phase4.md) for clean backbone training and unseen evaluation.

Phase 5 now includes verified evaluation reports, recognizer export and a saved-clip prediction command. See the [Phase 5 guide](docs/phase5.md) and [current evidence](reports/phase5_summary.md). The released PHOENIX-2014-T CorrNet checkpoint is available locally and passed a five-clip GPU extraction check. Full Head A and real A/B/C comparisons remain pending. Phase 4 uses generic ImageNet initialization and German fastText. The Windows cuDNN shutdown issue has a tested CUDA workaround in the Phase 5 guide. Historical checks are recorded in [Phase 2 status](reports/phase2_summary.md), [Phase 3 status](reports/phase3_summary.md), and [Phase 4 status](reports/phase4_summary.md).

Read [the setup guide](docs/setup.md) for commands, validation coverage, and limitations. The [beginner plan](SIGN_LANGUAGE_PROJECT_PLAN.md) explains the five phases. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) describes the original scaffold; its statements that all files are empty are historical and superseded by this README.

Quick start from PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m sign_language.cli.main prepare-data
```

Output: `data/prepared/phase1/`. Existing output is never overwritten; use a different `--output` for another version. The source dataset is read in place.

Tests:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_phase*.py' -v
```

Preparation has no third-party runtime dependencies. Phase 2 dependencies are available through `pip install -e ".[phase2]"`. Phase 1 dictionaries are project-owned; the Phase 2 importer provides the separate released CorrNet dictionary. See [experiment rules](docs/experiment_protocol.md) before training.
