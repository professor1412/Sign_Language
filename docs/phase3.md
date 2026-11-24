# Phase 3: text embeddings, Head B aur Head C

Phase 3 ka code implemented hai. Is phase mein hum sign ki pictures se aane wale features ko gloss ke text features ke saath compare karte hain. **Real PHOENIX comparison abhi pending hai:** pretrained CorrNet weights, real feature caches aur German fastText binary model chahiye. [Verification report](../reports/phase3_summary.md) mein completed checks hain.

## 1. Pehle flow samjho

```text
PNG frames -> frozen CorrNet -> saved video features
                                      |
                              temporal encoder
                                      |
                +---------------------+---------------------+
                |                     |                     |
             Head A                Head B                Head C
        learned classifier     video-text match     video-text match,
                                                   text adjustment ke saath
                |                     |                     |
                +---------------------+---------------------+
                                      |
                              CTC loss + Adam
                                      |
                         gloss sequence prediction

Gloss WETTER -> text wetter -> fastText -> 300 numbers -> B/C
```

**Embedding** word ko numbers ki list mein represent karta hai. German translation sentence yahan input nahi hai; har **gloss label** ka text input hai. Original gloss aur uska numeric ID unchanged rehte hain. Har experiment mein ek head select hota hai. Teenon heads ek saath train nahi hote.

| Head | Easy meaning | Config file in `configs/experiments/` |
|---|---|---|
| A | Har known label ke liye classifier seekhta hai | `head_a.yaml` |
| A-norm | A ke class vectors ki length normalize karta hai | `head_a_norm.yaml` |
| B | Video vector ko fixed fastText vectors se match karta hai | `head_b.yaml` |
| C | Ek shared learned layer se text vectors adjust karke match karta hai | `head_c.yaml` |
| B-rand | Text vectors ki jagah fixed random vectors | `head_b_random.yaml` |
| B-shuf | Ordinary text vectors ko labels ke beech shuffle karta hai | `head_b_shuffled.yaml` |

Controls se check hoga ki improvement text meaning ki wajah se hai ya arbitrary label vectors se bhi same result aata hai. Shuffle mein kuch rows apni original jagah reh sakti hain. Blank aur special marker rows random/shuffle operation ka part nahi hain.

## 2. Abhi run karne ke liye small check

Project folder mein PowerShell kholo:

```powershell
$env:PYTHONPATH = 'src'
python -m sign_language.cli.main smoke-test-phase3 --output runs/my_phase3_check --device cpu
```

Is command ko dataset, CorrNet weights ya fastText download ki zarurat nahi hai. Yeh invented features aur invented text vectors banata hai, six heads ko three epochs train karta hai, checkpoints reload/evaluate karta hai, aur B/C ko four sequences learn karata hai. Repeated signs bhi included hain. Three-epoch run ka kaam pipeline check karna hai; uska WER performance comparison nahi hai. Dummy train/dev patterns intentionally same hain.

Result: `runs/my_phase3_check/smoke_report.json`. Existing output folder overwrite nahi hota; dobara run ke liye naya naam do. GPU ke liye `--device cuda` use kar sakte ho. Is implementation ke recorded smoke results CPU ke hain.

## 3. Real experiment ke required inputs

| Required item | Expected location | Kyon chahiye? |
|---|---|---|
| Phase 1 vocabulary | `data/prepared/phase1/vocabularies/reference.json` | Same label IDs for every head |
| Train cache | `data/processed/corrnet_reference/train/` | Video features for learning |
| Dev cache | `data/processed/corrnet_reference/dev/` | Best checkpoint choose karna |
| German fastText model | `artifacts/pretrained/fasttext/cc.de.300.bin` | Gloss text ke 300-dimensional vectors |

Real caches banane ki commands [Phase 2 guide](phase2.md) mein hain. CorrNet checkpoint missing ho toh pehle woh requirement resolve karo; random CorrNet weights ke features ko pretrained features mat samjho.

Training uses the existing Phase 2 Python packages. Fresh environment mein:

```powershell
python -m pip install -e ".[phase2]"
```

Embedding table banane ke liye additionally:

```powershell
python -m pip install -e ".[text]"
```

`text` optional dependency fastText install karti hai. Windows/Python version ke hisaab se package build fail ho sakta hai; official instructions describe a C++11 compiler and Mac/Linux support. Agar native install fail ho, embedding bundle WSL/Linux mein generate karke Windows project mein use kar sakte ho. Bundle banne ke baad head training ko fastText package ki zarurat nahi. [Official installation and Python API](https://fasttext.cc/docs/en/python-module.html).

German **bin** model official [fastText model page](https://fasttext.cc/docs/en/crawl-vectors.html) se lo, archive extract karo, aur `cc.de.300.bin` ko table wale path par rakho. Folder banane ke liye `New-Item -ItemType Directory -Force artifacts/pretrained/fasttext` run karo. Model large hai aur load karne ke liye free RAM chahiye. `.vec` file ya compressed `.gz` ko `.bin` rename karna kaam nahi karega. Binary model unfamiliar strings ke liye subword vectors generate kar sakta hai.

## 4. Gloss embeddings banao aur inspect karo

```powershell
python -m sign_language.cli.main build-embeddings --fasttext-model artifacts/pretrained/fasttext/cc.de.300.bin
```

Output:

```text
data/text/embeddings/fasttext/reference/
  embeddings.pt  # Fixed vectors, vocabulary, source fingerprint, audit metadata
  audit.json     # Human-readable label-to-text mapping and similarity checks
```

`WETTER` becomes `wetter` for fastText, but prediction label remains `WETTER`. Ordinary vectors are normalized. Blank ID 0 aur `__...` marker labels ke learned vectors head ke andar bante hain; unko ordinary words nahi maana jaata.

`audit.json` mein inspect karo:

- `rows`: original ID, gloss aur normalized text.
- `normalization_collisions`: different labels jo same text ban gaye.
- `near_duplicate_pairs`: cosine similarity greater than 0.95 wale pairs.
- `source`: model filename aur SHA-256 fingerprint.

Invalid or zero ordinary vectors par command fail hoti hai. Similar vectors report hote hain, automatically delete nahi hote. Same text vectors wale different signs ko B distinguish nahi kar paayega; C ka shared transformation bhi identical input vectors ko separate nahi kar sakta.

Default mein `loc-`, `cl-`, `neg-` prefixes preserve hote hain, kyunki meaning badal sakta hai. Reviewed alternative ke liye `--strip-prefixes --output data/text/embeddings/fasttext/stripped` use karo, audit dekho, aur affected experiment configs mein new bundle path set karo. Different embedding bundles ko same controlled comparison mein mix mat karo.

## 5. A, B, C aur controls train karo

Head B example:

```powershell
python -m sign_language.cli.main train --config configs/experiments/head_b.yaml --train-cache data/processed/corrnet_reference/train --dev-cache data/processed/corrnet_reference/dev --output runs/known_signs/head_b_seed42 --seed 42
```

Head C ke liye `--config configs/experiments/head_c.yaml` aur output `runs/known_signs/head_c_seed42` use karo. A aur controls ke liye table mein diya config select karo. Current implementation ke saath A fresh train karo, taaki changed training code ko old Phase 2 run se mix na karo.

Default: 40 epochs, batch size 8, Adam learning rate 0.001, weight decay 0.001, cosine learning-rate schedule, seed 42. An epoch means selected training clips par ek complete pass. Every head uses the same temporal encoder settings. Same seed par encoder starting weights aur data-shuffle stream same rehte hain. Each head has its own weights; parameter counts `run.json` mein hain.

First-epoch check ke liye fresh run command mein `--stop-after-epochs 1` add karo. Us run ko continue karne ke liye same command se stop flag remove karke `--resume` add karo. Seed override diya tha toh resume mein bhi same `--seed` do. Resume last completed epoch se hota hai. Configuration, data, code aur environment match hone chahiye.

All six experiments for seed 42, sequentially:

```powershell
$heads = 'head_a', 'head_a_norm', 'head_b', 'head_c', 'head_b_random', 'head_b_shuffled'
foreach ($head in $heads) {
    python -m sign_language.cli.main train --config "configs/experiments/$head.yaml" --train-cache data/processed/corrnet_reference/train --dev-cache data/processed/corrnet_reference/dev --output "runs/known_signs/${head}_seed42" --seed 42
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $head" }
}
```

Choose individual commands or the loop for a fresh set of runs. Loop existing outputs overwrite nahi karega. Important comparison ko repeat karne ke liye seeds 43 aur 44 ke new output folders banao. Controls ka `control_seed: 42` fixed vector/shuffle assignment rakhta hai; training seed independently initialization aur batch order change karta hai. Control assignments bhi vary karna ho toh separate experiment family record karo.

Small real-data learning check ke liye B/C config ki copy banao, usmein debug learning rate/model size choose karo, aur:

```powershell
python -m sign_language.cli.main overfit --cache data/processed/corrnet_reference/train --config configs/experiments/head_b.yaml --limit 4 --max-steps 300 --output runs/head_b_real_overfit
```

`overfit` constant configured learning rate aur zero weight decay use karta hai. Four clips learn na hon toh report inspect karo; full training ko success assume mat karo. Real cache par learning check abhi execute nahi hua.

## 6. Results compare aur inspect karo

Full configured training budget complete hone ke baad:

```powershell
python -m sign_language.cli.main compare-runs --runs runs/known_signs/head_a_seed42 runs/known_signs/head_b_seed42 runs/known_signs/head_c_seed42 --output reports/tables/phase3_abc_seed42.json
```

Controls aur repeated seeds ke folders bhi `--runs` list mein add kar sakte ho. Each head ko same seed set, data, training settings, code/environment aur decoder chahiye. Text heads ke source embedding bundle bhi same hone chahiye. Partial-epoch-budget runs reject hote hain. Command lowest-dev-WER checkpoint per run choose karke average aur sample standard deviation save karti hai. One seed par standard deviation `null` hai, zero nahi.

Detailed Head B dev evaluation:

```powershell
python -m sign_language.cli.main evaluate --checkpoint runs/known_signs/head_b_seed42/best.pt --cache data/processed/corrnet_reference/dev --output runs/known_signs/head_b_seed42/dev_evaluation.json
```

| Output | Easy meaning |
|---|---|
| `run.json` | Settings, vocabulary, source fingerprints, parameter counts |
| `history.json` | Epoch-wise loss, dev WER, blank rate, B/C temperature |
| `best.pt` | Lowest dev WER wala checkpoint |
| `last.pt` | Latest completed epoch; resume ke liye |
| `best_dev_predictions.json` | Correct versus predicted gloss sequence per clip |
| `dev_evaluation.json` | WER, substitutions/deletions/insertions, blank rate, per-gloss recall, training-frequency bins |

Lower WER better hai. **Blank rate** valid time steps ka fraction hai jahan blank highest-score label tha; padding excluded hai. **Recall** batata hai ki reference mein kisi gloss ki occurrences mein se kitni correctly aligned/predicted thi. Frequency bins training counts se bante hain. Agar imported vocabulary mein counts nahi hain, frequency breakdown meaningful nahi hoga; use the Phase 1 vocabulary for this study.

Checkpoint ke andar fixed vectors aur learned weights save hote hain. Evaluation ke time original fastText model ya embedding bundle available hona required nahi. Resume mein existing source bundle badla mile toh command reject karti hai.

Use dev for model choices. Settings freeze hone ke baad hi [Phase 2's final test evaluation commands](phase2.md) follow karo. Current decoder greedy hai; beam search abhi implemented nahi. Full-data CorrNet features ka comparison known-sign study hai; clean unseen-sign claim Phase 4 ka separate work hai.

## 7. Code kahan hai?

| File under `src/sign_language/` | Responsibility |
|---|---|
| `text/normalization.py` | Gloss text cleaning; special-label rule |
| `text/fasttext_encoder.py` | Optional binary model loading and dimension check |
| `text/embedding_table.py` | ID-ordered vectors, validation and audit |
| `cli/commands/build_embeddings.py` | Embedding build command |
| `models/heads/text_similarity.py` | Head B, learned special vectors and temperature |
| `models/heads/projected_text.py` | Head C shared text projection |
| `models/heads/controls.py` | Fixed random/shuffled ordinary vectors |
| `models/heads/normalized_linear.py` | A-norm |
| `models/factory.py`, `models/recognizer.py` | Selected head construction and checkpoint loading |
| `training/trainer.py`, `training/loops.py` | Shared training, resume and diagnostics |
| `evaluation/alignment.py`, `evaluation/analysis.py` | Alignment-based recall and frequency bins |
| `evaluation/reporting.py` | Comparable dev-run summaries |
| `cli/commands/evaluate.py` | Saved-model evaluation |
| `cli/commands/smoke_phase3.py` | Repeatable dummy-data workflow |

Active experiment settings are the six `configs/experiments/head_*.yaml` files listed above. `configs/text/fasttext.yaml`, `configs/models/heads.yaml` and `configs/evaluation/known_signs.yaml` remain historical scaffold placeholders; these commands do not read them.

Checks:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_phase*.py' -v
```

`tests/test_phase3.py` covers ID preservation, cosine scores, gradient flow, fixed versus learned vectors, checkpoint reload, exact CPU resume, recall, embedding export and comparison restrictions. The real fastText binary loader has not been exercised locally because that model and package are absent.
