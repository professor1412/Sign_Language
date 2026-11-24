# Phase 4 code and checks

Recorded 2026-11-24. Code follows Phase 4 of the workspace five-phase plan. Research results are pending.

Implemented: frozen reduced-data study, generic-only CorrNet initialization, dual-CTC backbone training with gradient accumulation and epoch-boundary resume, clean feature extraction, clean head-training checks, inference-only vocabulary expansion for B/C and controls, and unseen evaluation.

The generated [study](../data/prepared/phase4/study.json) contains 2,809 train clips, 208 seen-only dev clips, all 519 dev/642 test clips, 668 seen glosses, 108 declared held-out glosses and 309 incidentally unseen glosses. It records the existing seeded pilot; it does not claim manual family balancing. Original dataset and Phase 1 outputs were not edited.

## Verification

- All 40 Phase 1-4 automated tests passed, including nine Phase 4 tests.
- Tests cover removal of entire held-out-containing clips, stable seen IDs, modified study detection, rejection of full-data cache provenance, expanded training candidates and non-seen dev selection.
- Small synthetic fixtures exercise fresh head training, B/C/control expansion, self-contained checkpoints and whole-split unseen evaluation. Learned parameters and old raw logits are preserved after expansion.
- Precision/recall tests include substitutions, insertions, missing support, incidental labels and natural OOV references.
- A small substitute backbone tests dual-CTC orchestration, partial gradient-accumulation groups, exact CPU resume and changed-frame detection. Clean extraction and cache resume are tested using a substitute model.
- Generic initializer tests reject an unrecognized checkpoint and verify 2-D to 3-D kernel inflation with mocked generic weights, keeping correlation parameters fresh.
- Actual CorrNet architecture passed one CPU forward/backward/Adam step with 16 random frames and random weights. Saved-feature logits matched direct logits exactly. [CPU graph report](../runs/phase4_checks/corrnet_graph_cpu.json).

The actual GPU graph also completed its numerical checks and wrote reports, but the Python process returned status 1 without a traceback on two attempts. This remains unresolved and is not counted as a fully successful GPU execution. A minimal CUDA tensor test exited normally. [GPU computation report](../runs/phase4_checks/corrnet_graph.json).

These are engineering checks on invented data, not PHOENIX accuracy measurements. The simplified clean training recipe uses sequence CTC + convolutional CTC with deterministic preprocessing, not the complete upstream VAC/distillation/augmentation recipe.

## Pending

- Generic ImageNet checkpoint: the download attempt was blocked by the environment's network socket restriction. The loader was tested with fixtures, not the actual binary.
- German fastText binary/package and audited real seen/expanded embedding bundles.
- Real clean backbone training, realistic full-clip memory validation, clean caches, complete head/control training, repeated seeds and final unseen measurements.
- GPU process-exit issue needs investigation before claiming validated CUDA training.

See [the beginner Phase 4 guide](../docs/phase4.md) for commands and output locations. No real recognition or semantic-generalization conclusion is available yet.
