# Original PKP2-seed13 epoch4

Authors: Xin Xu and Jun Yu, University of Science and Technology of China. Corresponding author: Jun Yu (harryjun@ustc.edu.cn).

One Function enters a canonical lexical code/comment extractor and structured pair packer (max 512 tokens, initial comment reservation 96). A shared two-layer CodeBERT with CLS linear heads returns independent Weakness/MAT logits. Legacy unused gates/pooler parameters remain to preserve the measured state dictionary; there is no deployed teacher, ensemble or active task gate.

Checkpoint identity, exact frozen policy and original training recipe are in `FROZEN_CONFIG.json`. PKP2 was initialized from supervised teacher layers [0,11] and a fresh head, then trained for four epochs with weighted BCE plus TRAIN-only teacher distillation. Historical teacher selection used internal validation. The default thresholds were fitted on public validation; results do not constitute untouched or hidden tests.

The original checkpoint is selected; head/ranking/FFN edits and the 50-forward experiment remain research diagnostics. Neither a different seed nor the grouped A13 student is a substitute. Weakness F1 is below the official baseline here; the reported deployment choice reflects the reference accuracy/cost formula. Near-clone overlap, prior selection exposure, lexical extraction errors and rare MAT positives limit conclusions.

No new model was run during repository preparation. Actual past T4 measurements, synthetic CLI tests and new-environment acceptance are distinct evidence. No hidden ranking or official wrapper acceptance is claimed. See README and `docs/RELEASE_STATUS.md`.
