# Phase 1 experiment rules

Input: ordered PHOENIX-2014-T frame sequences. Output: ordered glosses. Translation and live streaming are outside the core experiment.

Use official train/dev/test partitions for the known-sign control study. For the unseen-sign study, select held-out labels from training vocabulary only and remove whole training clips containing them. Train all sign-specific components on that reduced list; full-data reference features must not be reused for a clean unseen claim.

Select clean-study settings on the generated seen-only development subset. Keep final test results out of tuning. Train with seen candidates plus blank/special markers; append held-out text candidates at inference without weight updates. Track incidentally unseen and naturally out-of-training-vocabulary labels separately.

The generated seeded selection is a pilot. Review remaining data volume, sign families, and evaluation support before freezing the study. Any manually reviewed held-out list must be versioned and fixed before observing recognition outcomes.

Use the experiment register to record run ID, configuration, seed, model/checkpoint origin, manifest fingerprints, vocabulary, decoder, metrics, and artifact locations. Report WER and unseen precision/recall, controls, and seed variation; distinguish measured results from planned goals.
