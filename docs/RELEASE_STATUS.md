# Release status (2026-11-16)

Completed: named four-page paper with repository URL; MIT source license and upstream CodeBERT attribution; frozen original student/tokenizer/config release; original teacher and fixed split prerequisites; source-bound data preparation and fresh-head generation; TRAIN-only cache/student/tune implementations; ordered prediction export and saved-output evaluation; inference notebook; synthetic tests and source integrity checks.

Scope of verification: existing original PKP2 has recorded T4 deployment results. Repository tests/tokenizer parity and weights-only head reconstruction do not constitute a new complete training run. Fresh-cache recipe reproduction is explicitly separated from exact historic checkpoint regeneration. Earlier raw warm-start and Pair teacher retraining are not reconstructed; the fixed teacher is supplied. General CLI/new notebook execution on organizer hardware is not newly GPU-tested. Full fresh-environment training and organizer hidden evaluation/acceptance remain unverified.

Public source: https://github.com/XX123122/PairKD-NLBSE2027
Model and teacher prerequisites: https://github.com/XX123122/PairKD-NLBSE2027/releases/tag/v0.2.0
No competition form has been submitted. No real-model forward, training update, threshold fit or new public/hidden evaluation was executed in this closeout. The original checkpoint and float32 thresholds are unchanged.
