# Sign Language Project — Phase-wise File and Folder Guide

Is guide mein har phase ke liye dekh sakte ho ki **kaunsi file input hai, code kahan hai, aur result kahan milega**. Saare paths project ke main folder se start hote hain.

| Phase | Kaam | Abhi ki status |
|---|---|---|
| 1 | PNG frames aur labels ko map karke data ready karna | Code aur PHOENIX ke prepared files available hain. |
| 2 | CorrNet features nikalna aur Head A train karna | Code aur local checks available hain. Real run ke liye pretrained CorrNet weights chahiye. |
| 3 | Text embeddings ke saath Head B aur C banana aur compare karna | Code, six experiment configs aur synthetic checks available hain. Real fastText model aur CorrNet caches ke saath full comparison pending hai. |
| 4 | Training se hataaye hue signs par model test karna | Clean backbone, caches, vocabulary expansion aur unseen evaluation code implemented hai. Real training/results pending hain. |
| 5 | Final results, report, demo aur optional extensions | Report folders aur demo ka scaffold hai. Outputs final experiments se aayenge. |

Phase numbers [five-phase project plan](SIGN_LANGUAGE_PROJECT_PLAN.md) ke hain. Original PDF ki phase numbering alag hai. See [SLPlan formulas and methods](docs/slplan-formulas-and-methods.md).

## Phase 1 — Dataset ready karna

**Input** dataset banane walon ki original CSV aur PNG files hain. Har numbered PNG ek frame hai. Har CSV row batati hai ki clip ke correct signs kis order mein hain.

~~~text
phoenix-2014-T.v3/
└── PHOENIX-2014-T-release-v3/
    └── PHOENIX-2014-T/
        ├── annotations/manual/
        │   ├── PHOENIX-2014-T.train.corpus.csv   # Training ke correct labels
        │   ├── PHOENIX-2014-T.dev.corpus.csv     # Development ke correct labels
        │   └── PHOENIX-2014-T.test.corpus.csv    # Test ke correct labels
        └── features/fullFrame-210x260px/
            ├── train/<clip_id>/images*.png
            ├── dev/<clip_id>/images*.png
            └── test/<clip_id>/images*.png
~~~

**Code.** Command ka naam **prepare-data** hai. Phase 1 Python ki standard library use karta hai. Is command ki settings arguments se aati hain. **configs/data/phoenix2014t.yaml** aur **configs/data/held_out_split.yaml** future configuration ke placeholders hain.

~~~text
src/sign_language/
├── cli/
│   ├── main.py                      # prepare-data command register karta hai
│   └── commands/prepare_data.py      # Data check karta hai aur output bundle banata hai
├── data/
│   ├── annotations.py               # Original CSV se clip IDs aur labels padhta hai
│   ├── frame_reader.py              # PNG folders, frame numbers aur headers check karta hai
│   ├── manifest.py                  # Clip ka path, frame count aur labels ek record mein jodta hai
│   ├── validation.py                # Train, dev aur test ke clip IDs ko alag check karta hai
│   ├── vocabulary.py                # Labels ko numeric IDs deta hai, blank ka ID 0
│   └── splits.py                     # Selected held-out signs wali training clips hataata hai
└── core/provenance.py                # Data aur settings ke fingerprints banata hai
~~~

**Output.** Manifest ka matlab clip list hai. Har JSONL line ek clip ka record hai.

~~~text
data/prepared/phase1/
├── manifests/
│   ├── reference/
│   │   ├── train.jsonl               # Full training list
│   │   ├── dev.jsonl                 # Full development list
│   │   └── test.jsonl                # Full test list
│   └── reduced/
│       ├── train.jsonl               # Phase 4 ke liye reduced training list
│       ├── dev_seen_only.jsonl        # Sirf woh dev clips jinke labels reduced train mein hain
│       ├── dev_held_out.jsonl        # Dev clips jinmein selected held-out label aata hai
│       ├── test_seen_only.jsonl       # Sirf woh test clips jinke labels reduced train mein hain
│       └── test_held_out.jsonl       # Test clips jinmein selected held-out label aata hai
├── vocabularies/
│   ├── reference.json               # Full training label-to-ID mapping
│   └── reduced.json                  # Reduced training label-to-ID mapping
├── splits/held_out.json              # Held-out labels aur selection details
├── validation_report.json           # Clip/frame counts aur checks
└── experiment_register.json         # Experiment record rakhne ki jagah
~~~

**Tests aur notes.** **tests/test_phase1.py** data preparation verify karta hai. **reports/phase1_summary.md** execution results aur **reports/phase1_environment.json** Python/GPU check record karte hain. [Setup instructions](docs/setup.md) · [Experiment rules](docs/experiment_protocol.md)

**Agla phase kya lega?** Phase 2 full **reference** manifests aur vocabulary use karega. Phase 4 **reduced** manifests, vocabulary aur held-out list use karega.

## Phase 2 — Features cache karna aur Head A banana

**Input.** Phase 1 ke full manifests aur original PNGs se CorrNet features banenge. CorrNet ka released checkpoint apni original vocabulary ke saath load hota hai. Head A apni training ke liye Phase 1 wali vocabulary use karta hai. Dono ki label ID mappings alag hain.

~~~text
external/corrnet/upstream/                         # Download kiya hua official CorrNet source
├── modules/                                      # ResNet, TemporalConv aur BiLSTM code
└── preprocess/phoenix2014-T/gloss_dict.npy          # CorrNet ki original label mapping

artifacts/pretrained/corrnet/
├── vocabulary.json                               # Original dictionary ka imported JSON
└── phoenix2014t.pt                                # Pretrained weights, real run ke liye pending

data/prepared/phase1/
├── manifests/reference/{train,dev,test}.jsonl       # Kis clip ko process karna hai
└── vocabularies/reference.json                    # Head A ke target labels ki ID mapping

configs/training/
├── heads.yaml                                    # Full Head A training settings
└── overfit_debug.yaml                             # Small learning check ke settings
~~~

**Code.** CorrNet ke weights fixed rehte hain. Training mein temporal encoder aur Head A learn karte hain.

~~~text
src/sign_language/
├── cli/
│   ├── main.py                                    # Phase 2 commands ke arguments
│   └── commands/
│       ├── reference_vocabulary.py                # import-corrnet-vocabulary
│       ├── smoke.py                                # smoke-test, synthetic features par learning check
│       ├── overfit.py                              # overfit, kuch training clips par learning check
│       └── evaluate.py                             # evaluate, checkpoint ko score karta hai
├── core/
│   ├── settings.py                                 # Manifests, JSON aur vocabulary check
│   └── reproducibility.py                           # Seed, device aur resume states
├── data/
│   ├── transforms.py                               # PNGs ko CorrNet ke input mein convert karta hai
│   ├── dataset.py                                  # Cached features aur labels load karta hai
│   └── collate.py                                  # Unequal lengths wali clips ka padded batch
├── features/
│   ├── extractor.py                                # extract-features command ka actual code
│   ├── cache.py                                    # Feature files, metadata aur index save karta hai
│   └── validation.py                               # Saved features se prediction match check karta hai
├── models/
│   ├── backbones/corrnet.py                         # Official CorrNet model aur checkpoint load karta hai
│   ├── temporal/encoder.py                          # Features ko time order mein process karta hai
│   ├── heads/
│   │   ├── linear.py                                # Custom Head A
│   │   └── normalized_linear.py                      # Released CorrNet ka classifier
│   └── recognizer.py                                # Encoder aur Head A ko jodta hai
├── losses/ctc.py                                     # Correct label sequence se CTC loss
├── decoding/greedy.py                               # Scores se predicted sign sequence
├── evaluation/wer.py                                # Wrong, missing aur extra signs count karta hai
└── training/
    ├── trainer.py                                  # train command, config, Adam, dev WER aur resume
    ├── loops.py                                    # Batch training aur evaluation chalata hai
    └── checkpoints.py                              # Model aur training state save/load
~~~

Command ka actual code dhundte waqt **features/extractor.py** aur **training/trainer.py** kholo. Scaffold ke **cli/commands/extract_features.py** aur **cli/commands/train.py** abhi placeholders hain.

**configs/models/corrnet.yaml**, **configs/models/temporal.yaml** aur **configs/experiments/reference.yaml** future configuration ke placeholders hain. Current training CLI **configs/training/heads.yaml** padhti hai. Real feature cache ki guide mein location **data/processed/corrnet_reference/** hai. Scaffold ka **artifacts/features/reference/** separate reserved folder hai.

**Real run ke output paths** [Phase 2 run guide](docs/phase2.md) mein use kiye gaye hain. Ye files relevant commands chalne ke baad banengi.

~~~text
data/processed/corrnet_reference/
├── dev_first5/                         # Pehle 5 videos ki extraction check
├── train/                              # Full training feature cache
├── dev/                                # Full development feature cache
└── test/                               # Final test feature cache

Har cache folder ke andar:
├── tensors/<video-hash>.pt              # Har clip ke actual feature numbers
├── metadata.json                        # Model aur preprocessing details
├── index.json                           # Clip list, actual lengths aur feature paths
└── reference_report.json                # CorrNet predictions, WER aur checks

runs/
├── real_clip_overfit/
│   ├── overfit.pt                       # Small-set check ka saved model
│   └── overfit_report.json              # Sahi aur predicted labels
└── head_a_seed42/
    ├── run.json                         # Settings, vocabulary aur input fingerprints
    ├── history.json                     # Har epoch ka loss aur dev WER
    ├── best.pt                          # Dev WER se selected model
    ├── last.pt                          # Last completed epoch, resume ke liye
    ├── best_dev_predictions.json        # Best model ki dev predictions
    ├── dev_evaluation.json               # Separate dev evaluation run ka example output
    └── test_evaluation.json              # Separate final test run ka example output
~~~

**Existing checks.** **tests/test_phase2.py** cache, CTC, decoding, padding aur resume test karta hai. **runs/phase2_smoke/**, **runs/phase2_adapter_check/**, **runs/phase2_cli_check/** aur **runs/phase2_overfit_cli/** local check results rakhte hain. Real accuracy abhi measure karni hai. [Phase 2 verification](reports/phase2_summary.md)

**Agla phase kya lega?** Phase 3 same cached features, temporal encoder, CTC pipeline aur Head A comparison ko use karega.

## Phase 3 — Head B aur Head C ke saath text ka effect test karna

**Inputs.** Phase 2 ke train/dev feature caches aur full training vocabulary. German fastText model ka guide path **artifacts/pretrained/fasttext/cc.de.300.bin** hai. Label ka original ID wahi rahega. Text embedding ke liye ek clean version use hoga, jaise **WETTER → wetter**. Fair comparison ke liye Head A bhi current training code se fresh train hota hai.

**Code implemented hai.** Naye heads har time step par signs ke scores dete hain. Shared training aur WER pipeline reuse hoti hai. Commands aur easy explanation [Phase 3 guide](docs/phase3.md) mein hain.

~~~text
src/sign_language/
├── cli/commands/build_embeddings.py   # Vocabulary se text-vector table banane ki command
├── text/
│   ├── normalization.py               # Embedding ke liye gloss text clean karta hai
│   ├── fasttext_encoder.py             # German fastText se vector nikalta hai
│   └── embedding_table.py              # Labeled vectors ko gloss ID order mein rakhta hai
├── models/
│   ├── factory.py                      # Config ke hisaab se selected head banana
│   └── heads/
│       ├── text_similarity.py          # Head B, fixed text vectors se match
│       ├── projected_text.py            # Head C, shared text adjustment ke baad match
│       └── controls.py                 # Random aur shuffled vector checks
├── evaluation/alignment.py             # Prediction aur reference ko align karta hai
├── evaluation/analysis.py              # Per-gloss recall aur training-frequency bins
├── evaluation/reporting.py             # Matching runs ki mean/std dev-WER comparison
└── cli/commands/smoke_phase3.py         # Six heads ka dummy-data check

configs/
├── text/fasttext.yaml                   # Placeholder; commands do not read this
├── models/heads.yaml                    # Placeholder; commands do not read this
├── evaluation/known_signs.yaml          # Placeholder; commands do not read this
└── experiments/
    ├── head_a.yaml                      # A ka comparable baseline
    ├── head_a_norm.yaml                 # Optional normalized linear control
    ├── head_b.yaml                      # Fixed text vectors ka experiment
    ├── head_c.yaml                      # Shared text projection ka experiment
    ├── head_b_random.yaml               # Random vectors ka control
    └── head_b_shuffled.yaml             # Shuffled fastText rows ka control
~~~

Head A **models/heads/linear.py** mein aur A-norm **models/heads/normalized_linear.py** mein hai. **models/recognizer.py**, **training/trainer.py**, **training/loops.py**, **cli/main.py**, **cli/commands/overfit.py** aur **cli/commands/evaluate.py** ab all six heads support karte hain. Tests **tests/test_phase3.py** mein hain. **decoding/beam.py** abhi planned hai; current decoding greedy hai.

**Commands se banne wale outputs.** Real-data outputs pending hain; dummy check **runs/phase3_smoke/** mein hai. [Verification report](reports/phase3_summary.md) current status explain karta hai.

~~~text
data/text/embeddings/fasttext/reference/     # embeddings.pt aur audit.json
runs/known_signs/<head>_seed<seed>/           # Har A/B/C ya control run ka model, settings aur WER
reports/tables/                             # Sabhi heads ki dev WER comparison table
reports/figures/                            # Optional per-video alignment plots
~~~

**Agla phase kya lega?** Phase 4 B/C ka head design aur text embedding code reuse karega. Us study ke liye naye backbone aur head weights reduced data par seekhe jaayenge.

## Phase 4 — Unseen signs ka test

**Inputs.** Phase 1 ke manifests aur held-out list se **data/prepared/phase4/study.json** create ho gaya hai. Ismein 2,809 reduced train clips, 208 seen-only dev clips, complete dev/test lists aur fixed candidate IDs hain. **seen_vocabulary.json** mein 668 glosses aur blank hain; **expanded_vocabulary.json** mein 108 declared held-out glosses append hote hain. Existing pilot selection freeze hui hai; manual family-balancing ka claim nahi hai.

**Code aur model inputs.** Generic ResNet18 initialization ke saath fresh CorrNet sirf allowed clips par train hota hai. Full-data Phase 2 caches clean study mein reject hote hain. Commands aur requirements [Phase 4 guide](docs/phase4.md) mein hain.

~~~text
artifacts/pretrained/generic/                 # Required resnet18-f37072fd.pth
external/corrnet/upstream/                   # CorrNet architecture aur training reference
external/s3d/                                # Optional S3D alternative ka source

src/sign_language/
├── models/backbones/
│   ├── corrnet.py                           # Shared CorrNet architecture adapter
│   ├── clean_corrnet.py                     # Generic initialization and clean checkpoint loading
│   ├── base.py                              # Shared backbone interface, planned
│   └── s3d.py                               # Optional S3D adapter, planned
├── inference/vocabulary_expansion.py        # Test par held-out labels ki class rows add karna
├── data/unseen_protocol.py                  # Study freeze aur leakage checks
├── training/clean_backbone.py               # Dual-CTC training, selection and resume
├── features/clean_extractor.py              # Clean cache extraction and checks
├── evaluation/
│   ├── alignment.py                        # Predicted sign ko reference sign se align karna
│   └── unseen.py                           # Unseen recall, precision aur group-wise results
└── text/
    ├── sentence_encoder.py                 # Optional alternate text embeddings
    └── descriptions.py                     # Optional handshape/location/movement descriptions

configs/
├── training/backbone.yaml                   # Reduced split backbone training settings
├── models/s3d.yaml                          # Optional S3D settings
├── text/alternatives.yaml                   # Optional alternate text encoders
├── experiments/
│   ├── clean_head_*.yaml                    # Active A/B/C/control training configs
│   ├── clean_backbone.yaml                  # Historical placeholder, not loaded
│   └── unseen_signs.yaml                    # Historical placeholder, not loaded
└── evaluation/unseen_signs.yaml             # Historical placeholder, not loaded
~~~

**features/cache.py**, **training/trainer.py**, Phase 3 heads aur **cli/main.py** shared hain. Tests **tests/test_phase4.py** mein aur actual CorrNet graph check **tests/manual/check_clean_corrnet.py** mein hai. Optional S3D, descriptions aur alternate text encoders abhi placeholders hain.

**Outputs.** Frozen study aur debug checks available hain; real trained models/caches pending hain. See [Phase 4 verification](reports/phase4_summary.md).

~~~text
data/prepared/phase4/                        # Frozen study, manifests and vocabularies
runs/clean_backbone_seed42/                  # Fresh video encoder, best.pt and last.pt
data/processed/clean_phase4/                 # Separate train/dev_seen/dev/test caches
data/text/embeddings/fasttext/clean_seen/     # Seen-only embedding bundle and audit
data/text/embeddings/fasttext/clean_expanded/ # Seen-plus-held-out bundle and audit
runs/clean_heads/<head>_seed<seed>/           # Head weights, expanded.pt and unseen metrics
runs/phase4_checks/                          # Random-frame graph-check reports
reports/tables/                              # Familiar aur held-out signs ka comparison
~~~

**Agla phase kya lega?** Phase 5 ko selected models, exact split/candidate lists, final predictions aur metrics milenge.

## Phase 5 — Report, demo aur optional extensions

**Inputs.** Phase 2–4 ke **run.json**, **history.json**, selected checkpoints aur saved predictions. Published reference WER aur local measured WER ko report mein clearly label karo.

Official PHOENIX benchmark se compare karne ke liye original dataset ke **evaluation/sign-recognition/** folder mein dev/test **.stm** answer keys, **.ctm** example predictions aur **evaluatePHOENIX-2014-T-signrecognition.sh** scorer hain. Phase 2 ki JSON evaluation project ka raw-gloss WER report karti hai. [Dataset evaluation instructions](phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/evaluation/README)

**Planned code aur results.**

~~~text
src/sign_language/
├── evaluation/
│   ├── analysis.py                           # Wrong, missing aur extra signs ke examples
│   └── reporting.py                          # Metrics se final tables banana
├── inference/
│   ├── predictor.py                          # Saved model se demo prediction
│   ├── validation.py                         # Demo input aur model files check
│   └── export.py                             # Model, vocabulary aur settings bundle karna
└── cli/commands/
    ├── predict.py                            # Demo run karne ki planned command
    └── export.py                             # Model package banane ki planned command

configs/inference/local.yaml                  # Local demo settings
tests/smoke/test_prediction_bundle.py        # Exported model ke demo check ki jagah
artifacts/exports/                            # Shareable model, vocabulary aur preprocessing settings
reports/
├── tables/                                   # Final WER aur recall comparison tables
├── figures/                                  # Charts aur prediction/alignment plots
├── error_analysis/                           # Sahi aur galat predictions ke samjhaaye hue examples
└── final/                                    # Final report aur experiment summary
docs/
├── architecture.md                           # Model aur data flow
├── data_contracts.md                         # Inputs, feature shapes aur output formats
├── evaluation.md                             # Final scores kaise nikle
├── inference.md                              # Demo kaise use karein
├── model_card.md                             # Model kis data par bana aur kahan weak hai
└── troubleshooting.md                        # Common errors aur fixes
~~~

**Optional research files.** Spiking experiment ke liye **models/backbones/spiking.py**, **configs/models/spiking.yaml**, **configs/experiments/spiking.yaml** aur **external/spiking/** hain. Pooled video se signs ki list predict karne wale superposition experiment ke liye **losses/auxiliary.py** aur **configs/experiments/superposition.yaml** hain. Dono ke results **runs/extensions/** mein ja sakte hain.

**Optional service files.** Local demo ko API banane ka planned code **src/sign_language/service/** ke andar hai, jisme **routes/** requests handle karega. **configs/inference/service.yaml**, **tests/integration/test_service.py**, **deployment/docker/**, **deployment/monitoring/** aur **docs/deployment.md** us kaam ke liye hain.

## Sab phases ke shared files

| File ya folder | Use |
|---|---|
| [README.md](README.md) | Project start karne aur existing phases navigate karne ke liye. |
| [pyproject.toml](pyproject.toml) | Python version, package settings, command entry point aur Phase 2 dependencies. |
| [SIGN_LANGUAGE_PROJECT_PLAN.md](SIGN_LANGUAGE_PROJECT_PLAN.md) | Har phase ka goal aur completion checklist. |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Original scaffold ka full tree, jisme optional aur blank files bhi listed hain. |
| [data/README.md](data/README.md) | Data storage conventions. Working Phase 1 outputs **data/prepared/phase1/** mein hain. |
| **.gitignore**, **.dockerignore** | Dataset aur large outputs ko Git/container build context se bahar rakhte hain. |
| **\_\_init\_\_.py**, **.gitkeep** | Python package markers aur empty folder placeholders. Diagram mein repeat hone wale markers omit kiye hain. |

Yaad rakhne ke liye **src/** mein code, **configs/** mein settings, **data/** mein clip lists aur feature caches, **runs/** mein har run ke results, aur **reports/** mein final comparison milta hai.
