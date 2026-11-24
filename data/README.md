# Where the data belongs

The existing dataset remains outside this `data/` folder, at this path relative to the project root:

```text
phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
```

Set that location as the dataset root when you later implement `configs/paths.yaml` or your local environment configuration. It is documented here rather than configured now, because configuration files are intentionally empty.

Observed source layout:

```text
PHOENIX-2014-T/
  annotations/manual/
    PHOENIX-2014-T.train.corpus.csv
    PHOENIX-2014-T.train-complex-annotation.corpus.csv
    PHOENIX-2014-T.dev.corpus.csv
    PHOENIX-2014-T.test.corpus.csv
  features/fullFrame-210x260px/
    train/
    dev/
    test/
  evaluation/
    sign-recognition/
    sign-translation/
  models/languagemodels/
```

The dataset directory named `features/fullFrame-210x260px` holds the provided frame sequences. It is distinct from the learned numerical feature caches your project will generate under `artifacts/features/`.

Use the standard corpus annotations for the initial recognition setup and inspect the annotation schema before implementation. Keep gloss targets separate from translated German sentences. Choose special annotations or language-model decoding only through an explicit experiment setting.

Do not rewrite the original dataset to create an experiment split. Write lists of allowed video IDs under `data/manifests/` and `data/splits/`. Filtering a list avoids another 40 GB copy.

`reference` folders are for the ordinary full-data control study. `reduced` folders are for the clean training split that excludes held-out signs. Always record the split, vocabulary, preprocessing, and source checkpoint together.

Small manifests, split lists, and dictionaries may be versioned after reviewing dataset terms and sensitive fields. Large frame collections, weights, features, embeddings, uploads, and runs are excluded by the supplied `.gitignore`. Do not force-add the dataset.

The setup only checked directory and annotation filenames. Validate frame completeness and annotation/video correspondence when implementing Phase 1.
