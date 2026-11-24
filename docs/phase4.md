# Phase 4: training se bahar rakhe signs ko test karna

Phase 4 ka **code implemented hai**. Full PHOENIX training aur unseen-sign accuracy abhi measure nahi hui. Is guide ke commands project root se PowerShell mein run karo. [Verification and remaining work](../reports/phase4_summary.md).

## 1. Is phase mein kya badlega?

```text
Phase 1 ka fixed held-out label list
                 |
Un labels wali poori training clips remove
                 |
Generic ImageNet weights -> fresh CorrNet training on remaining clips
                 |
Clean video feature caches
                 |
Fresh temporal encoder + A / B / C (only seen labels during training)
                 |
Weights freeze -> B/C mein all held-out text candidates append
                 |
Same complete evaluation split -> WER, unseen recall, precision, false guesses
```

Example: training mein `RAIN` aur `SUN` allowed hain, `SNOW` held-out hai. `SNOW` wali poori training clip remove hogi. Training complete hone ke baad B/C ki answer list mein `SNOW` ka text vector add hoga. **Naya answer add karna aur usko correctly recognize karna alag cheezein hain.** Accuracy experiment se pata chalegi.

Phase 2 ka released CorrNet full training set dekh chuka tha. Uske weights ya feature caches clean unseen study mein use nahi hote. Phase 4 mein upstream CorrNet **architecture/code** reuse hota hai, aur initialization generic ImageNet ResNet18 weights se hota hai. Phase 2 ka released sign checkpoint is workflow ke liye required nahi hai.

## 2. Frozen study files

Current Phase 1 outputs se study already create ki gayi hai:

```text
data/prepared/phase4/
  study.json                 # Frozen lists, candidate rules and fingerprint
  seen_vocabulary.json       # 668 seen glosses + CTC blank
  expanded_vocabulary.json   # Same IDs + 108 held-out glosses appended
  train.jsonl                # 2,809 allowed training clips
  dev_seen.jsonl             # 208 seen-only dev clips for model selection
  dev.jsonl                  # All 519 official dev clips
  test.jsonl                 # All 642 official test clips
```

There are also 309 **incidentally unseen** labels: unki clips held-out labels ke saath remove hone se woh bhi training mein nahi bache. Unko declared 108 held-out labels ke saath mix nahi karte. **Natural OOV** means original full training vocabulary mein bhi absent label. Both categories remain in references and error counts; they are not added as candidates in this study.

Fresh workspace mein study banane ka command:

```powershell
$env:PYTHONPATH = 'src'
python -m sign_language.cli.main prepare-unseen-study
```

Current folder par command dobara run karoge toh overwrite refuse karega. Another version ke liye `--output data/prepared/phase4_v2` use karo, phir later commands mein matching `--study` path do. Original dataset aur Phase 1 files change nahi hote.

This version freezes the existing seeded **pilot** selection. Manual sign-family balancing/review ka claim nahi hai. Scientific study ke liye list ko outcomes dekhne se pehle review karo; change required ho toh versioned Phase 1/4 artifacts se new experiment start karo. Poor results dekhkar hidden labels swap mat karo.

## 3. Environment aur generic weights

```powershell
$env:PYTHONPATH = 'src'
python -m pip install -e ".[phase2]"
```

Same torch/NumPy/OpenCV/PyYAML packages use hote hain. Text embeddings banane ke liye optional fastText setup [Phase 3 guide](phase3.md) mein hai. German `cc.de.300.bin` abhi missing hai.

Generic ResNet18 checkpoint approximately 45 MB hai. Official [Torchvision ResNet18 weight definition](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) uses the URL below. Agent's download attempt was blocked by the environment's network restriction; file abhi locally available nahi hai. Apne normal terminal mein download kar sakte ho:

```powershell
New-Item -ItemType Directory -Force artifacts/pretrained/generic
if (-not (Test-Path artifacts/pretrained/generic/resnet18-f37072fd.pth)) {
    Invoke-WebRequest -Uri https://download.pytorch.org/models/resnet18-f37072fd.pth -OutFile artifacts/pretrained/generic/resnet18-f37072fd.pth
}
```

Loader official filename's SHA-256 prefix `f37072fd` verify karta hai aur full digest record karta hai. Same name se koi sign-trained checkpoint rename karke use nahi kar sakte. Spatial ResNet weights load hote hain; correlation modules, temporal layers aur classifiers freshly initialize hote hain.

## 4. Clean CorrNet train karo

```powershell
python -m sign_language.cli.main train-clean-backbone --output runs/clean_backbone_seed42 --device cuda
```

Settings: `configs/training/backbone.yaml`. Default 40 epochs, Adam learning rate 0.0001, weight decay 0.0001, gradient clipping 5, one complete clip per forward pass, and eight clips accumulated per optimizer update. Last smaller group is averaged correctly.

**Training method:** sequence CTC plus convolutional CTC, equal weights by default. Both losses train the fresh CorrNet. This is a documented project starting recipe; it does not reproduce all upstream VAC distillation losses or augmentation. Preprocessing uses the existing deterministic resize/crop and temporal padding from Phase 2. Original sequence order and full targets remain intact.

One-epoch check for a fresh run:

```powershell
python -m sign_language.cli.main train-clean-backbone --output runs/clean_backbone_seed42 --device cuda --stop-after-epochs 1
```

Choose the full command or the one-epoch command for a new folder. Continue the interrupted run with:

```powershell
python -m sign_language.cli.main train-clean-backbone --output runs/clean_backbone_seed42 --device cuda --resume
```

Resume restores model, optimizer, scheduler, shuffle/random states, and frame fingerprints at the last completed epoch. Same study, code, config and environment are required. Previously read frames changing between epochs/resume cause an error. Model selection uses only `dev_seen`; test clips never train the backbone or choose `best.pt`.

Outputs: `run.json`, `history.json`, `last.pt`, `best.pt`, `best_dev_predictions.json`. Pick the final dev-selected `best.pt` before building downstream caches.

**Memory:** full-video backpropagation is much heavier than Phase 2 feature extraction. A short random-frame check does not establish that long PHOENIX clips fit an 8 GB GPU. Gradient accumulation reduces clips per forward pass; it cannot reduce a single clip's memory. No frame truncation, skipped clips, mixed precision, or activation checkpointing is silently applied. CPU is supported with `--device cpu`; a GPU with more memory may be needed for practical full training.

## 5. Clean feature caches banao

Use the same selected checkpoint and extraction device for every cache:

```powershell
foreach ($split in 'train', 'dev_seen', 'dev') {
    python -m sign_language.cli.main extract-clean-features --checkpoint runs/clean_backbone_seed42/best.pt --split $split --output "data/processed/clean_phase4/$split" --device cpu
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $split" }
}
```

`--device cuda` bhi supported hai; local GPU caveat below applies. Existing identical cache resume karne ke liye individual command mein `--resume` add karo. Metadata changed ho toh new folder use karo. First three available clips par cached-feature logits original-model logits se compare hote hain.

Cache records include the clean study, checkpoint, generic initialization, source code, preprocessing and manifest fingerprints. New backbone checkpoint select kiya toh new caches banao. Phase 2 caches ko rename karne se woh clean nahi banenge.

## 6. Seen-label text embeddings aur fresh heads

First build text vectors **only for seen training candidates**:

```powershell
python -m sign_language.cli.main build-embeddings --vocabulary data/prepared/phase4/seen_vocabulary.json --fasttext-model artifacts/pretrained/fasttext/cc.de.300.bin --output data/text/embeddings/fasttext/clean_seen
```

Audit inspect karo. Training vocabulary aur vector order exactly match hone chahiye. Original glosses/IDs unchanged hain.

Train six configurations sequentially:

```powershell
$heads = 'head_a', 'head_a_norm', 'head_b', 'head_c', 'head_b_random', 'head_b_shuffled'
foreach ($head in $heads) {
    python -m sign_language.cli.main train --config "configs/experiments/clean_$head.yaml" --vocabulary data/prepared/phase4/seen_vocabulary.json --train-cache data/processed/clean_phase4/train --dev-cache data/processed/clean_phase4/dev_seen --output "runs/clean_heads/${head}_seed42" --seed 42
    if ($LASTEXITCODE -ne 0) { throw "Head training failed: $head" }
}
```

Head A/B/C architecture aur optimizer Phase 3 wale hain, but all weights freshly train hote hain. `train` detects clean caches and enforces exactly the frozen reduced train list, seen-only dev list and seen vocabulary. Expanded candidate vocabulary during training reject hoti hai.

Same seeds aur settings par comparison command use kar sakte ho:

```powershell
python -m sign_language.cli.main compare-runs --runs runs/clean_heads/head_a_seed42 runs/clean_heads/head_b_seed42 runs/clean_heads/head_c_seed42 --output reports/tables/phase4_seen_dev.json
```

Repeat important comparisons with seeds 43/44 using new folders. Reusing one clean backbone across head seeds measures head-training variability; it does not measure variability from retraining the whole backbone.

## 7. Weights freeze karke unseen candidates add karo

First make the expanded text table with the **same fastText binary and normalization**:

```powershell
python -m sign_language.cli.main build-embeddings --vocabulary data/prepared/phase4/expanded_vocabulary.json --fasttext-model artifacts/pretrained/fasttext/cc.de.300.bin --output data/text/embeddings/fasttext/clean_expanded
```

Head B example:

```powershell
python -m sign_language.cli.main expand-vocabulary --checkpoint runs/clean_heads/head_b_seed42/best.pt --embeddings data/text/embeddings/fasttext/clean_expanded/embeddings.pt --output runs/clean_heads/head_b_seed42/expanded.pt
```

Head C ke liye `head_b_seed42` ko `head_c_seed42` se replace karo. C applies its already learned shared projection to new text rows. No optimizer step happens. Learned weights, blank/special vectors, temperature, old candidate IDs and old raw scores remain unchanged. New competitors can still change softmax probabilities and final predictions.

The command appends all 108 declared labels for every video; correct labels dekhkar per-video shortlist nahi banati. Source fastText identity, text normalization aur original seen vectors training se match hone chahiye. Expanded checkpoint inference-only hai; ismein optimizer/resume state nahi hota.

A/A-norm seen-only baselines rehte hain: new class weights bina learning create nahi karte. B-rand/B-shuf support expansion too. For controls, existing trained rows stay unchanged; new held-out rows are randomized or shuffled within the new-label set, using `control_seed + 1`. This separate control rule is saved in the expanded checkpoint. Incidental/natural OOV labels remain outside the candidate set and remain in evaluation references.

## 8. Unseen evaluation aur honest comparison

Dev diagnostic example:

```powershell
python -m sign_language.cli.main evaluate-unseen --checkpoint runs/clean_heads/head_b_seed42/expanded.pt --cache data/processed/clean_phase4/dev --split dev --output runs/clean_heads/head_b_seed42/dev_unseen.json
```

Checkpoint/settings selection continues to use **seen-only dev**. Full dev with held-out labels is a diagnostic; using those outcomes to retune would change the declared study. Before looking at final test, freeze choices.

Final test cache and evaluation:

```powershell
python -m sign_language.cli.main extract-clean-features --checkpoint runs/clean_backbone_seed42/best.pt --split test --output data/processed/clean_phase4/test --device cpu
python -m sign_language.cli.main evaluate-unseen --checkpoint runs/clean_heads/head_b_seed42/expanded.pt --cache data/processed/clean_phase4/test --split test --output runs/clean_heads/head_b_seed42/test_unseen.json
```

Use the same command for C/controls. For A, pass its `best.pt` without expansion. Evaluate B/C's original `best.pt` on the same full split as well to measure how adding candidates affects familiar signs. Report files must use different names.

| Result | Easy meaning |
|---|---|
| Overall WER | All sequence mistakes, including unknown reference labels |
| Seen-only WER | Extra candidates ne familiar clips ko kitna affect kiya |
| Contains-held-out WER | Unseen labels wali sentences par errors |
| Held-out recall | Actual unseen occurrences mein se kitni correctly recognized |
| Held-out precision | Predicted unseen occurrences mein se kitni correct |
| False positives | Wrong unseen guesses, including insertions/substitutions |
| Per-gloss counts | Support, correct predictions, precision/recall for each held-out sign |
| Incidental/natural-OOV groups | Other missing labels ka separate effect |

Correct matches full-sequence minimum-edit alignment se count hote hain. Reference support zero ho toh recall `null`; predicted count zero ho toh precision `null`. Zero denominators ko fake zero accuracy nahi banaya jaata. Seen-only A ki held-out recall supported labels par zero hogi. B/C ka zero se better hona alone semantic benefit prove nahi karta; controls aur seen-sign damage bhi compare karo.

Clip groups such as contains-held-out and contains-natural-OOV overlap kar sakte hain. Evaluation requires the whole declared split from the same clean backbone; filtered easy clips silently accept nahi hote. Current decoding greedy and scoring raw gloss WER hai, official sclite cleanup nahi.

## 9. Code aur verification

| File under `src/sign_language/` | Responsibility |
|---|---|
| `data/unseen_protocol.py` | Frozen lists/candidates, fingerprints and leakage checks |
| `models/backbones/clean_corrnet.py` | Generic initialization and clean checkpoint loading |
| `training/clean_backbone.py` | Dual-CTC backbone training, seen-dev selection and resume |
| `features/clean_extractor.py` | Separate clean caches and feature equivalence checks |
| `training/trainer.py` | Fresh heads with clean cache/candidate enforcement |
| `inference/vocabulary_expansion.py` | Inference-only B/C and control expansion |
| `evaluation/unseen.py` | Unseen group WER, precision, recall and false guesses |
| `cli/main.py` | New commands |

`configs/training/backbone.yaml` and six `configs/experiments/clean_head_*.yaml` files are active settings. Other earlier scaffold files, including `configs/experiments/clean_backbone.yaml` and `unseen_signs.yaml`, are not loaded by these commands.

Run the automated checks without downloading model weights:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_phase*.py' -v
```

Optional actual CorrNet graph check with random pixels and random weights:

```powershell
python tests/manual/check_clean_corrnet.py --device cpu --output runs/my_phase4_check/corrnet_graph.json
```

The CPU graph check passed. On the local GPU, forward/backward, optimizer update and cache-equivalence assertions completed and wrote a passing report, but the Python process exited with status 1 without a traceback. GPU execution is therefore not fully validated; investigate that environment/runtime issue before a long CUDA run. A full real-data training run, generic binary import, fastText import and unseen accuracy measurement remain pending. Optional phrase encoders and sign descriptions are later experiments.
