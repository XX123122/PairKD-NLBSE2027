# Reproduction scope and required inputs

## Reuse the measured checkpoint

`scripts/import_model_zip.py` accepts only the exact frozen deployment ZIP, verifies the checkpoint before import, and prepares the asset layout used by `cli.py`. No training labels or teacher are required for inference. New predictions can be evaluated separately using `scripts/evaluate_saved.py`; gold rows must have exactly `index` and `Label` (two binary values) in identical order to saved predictions.

## Reproduce the student recipe from the supplied teacher

The public release supplies the exact original Pair+CLS42 teacher (SHA 447baa87fd51a5e3519c6662f60d2a42e5e88aebcf42ac84441559b418110c0f) and fixed split (SHA f0e361f9bf741dd8f6c81137e9977d5a28332c0949627f0d407cc067de0eafe9), separately from inference assets. Obtain PairKD-TRAIN-prerequisites-v0.2.0.zip from the release and verify its accompanying SHA256. Unpack it into `training_assets/`.

Obtain the unmodified `train` split of NLBSE/nlbse27-code-comment-classification (214,586 rows). The preparation snapshot records dataset revision feca19b90d1cc9fcf8ba5d6e693c4a94e249585c. One explicit example in the training environment:

```python
from datasets import load_dataset
frame = load_dataset('NLBSE/nlbse27-code-comment-classification', revision='feca19b90d1cc9fcf8ba5d6e693c4a94e249585c', split='train')
frame.to_parquet('official_train.parquet')
```

Use the pinned training dependencies, including the appropriate PyTorch build. Preparing data/head executes no encoder forward, but is a real preprocessing job:

```bash
python scripts/prepare_reproduction.py --train-parquet official_train.parquet --reproduction-assets training_assets --output reproduction
python cli.py cache --job reproduction/cache_job.json
```

Inspect `cache_job.json` and set `explicit_future_execution_consent` to true, then explicitly run `python cli.py cache --job reproduction/cache_job.json --execute`. This generates raw teacher logits for TRAIN only. After successful cache completion:

```bash
python scripts/bind_generated_cache.py --prepared reproduction
python cli.py train --job reproduction/train_job.json
```

Inspect `train_job.json`, set its consent to true, then run `python cli.py train --job reproduction/train_job.json --execute`. These commands are explicit instructions for a reproducer; they were not executed as a training run during release preparation. Original student supervision, sample orders, update count, layer map and optimization are retained. New generated head/template and teacher-cache hashes are recorded, not asserted identical to the historical files. PyTorch-dependent initialization may differ even with the same seed; the measured deployment checkpoint remains separately available unchanged. The portable trainer uses workers0 rather than original workers4 and is a recipe reproduction, not bit-identical historical checkpoint regeneration.

Original unpublished teacher logits and fresh head are no longer mandatory for this fresh-cache route. The supplied teacher is a fixed prerequisite; recreating its earlier raw warm-start/teacher-selection campaign from generic CodeBERT is not implemented by these student commands. This dependency boundary is deliberate and documented. Full new-environment end-to-end retraining has not been independently rerun; CPU/tokenizer/contract checks establish a narrower scope.

## Evaluate and tune

`scripts/predict_and_export.py` runs one selected/reproduced endpoint against supplied official Parquet rows (Function and Label), writes ordered probabilities and gold separately, and produces metrics without threshold search. `--candidate-assets` identifies a bootstrapped asset layout. Its default checkpoint binding remains original PKP2. Reproduced checkpoints need an explicitly separate manifest/identity; never silently replace the selected checkpoint.

`scripts/evaluate_saved.py` reports frozen and fixed-0.5 policies from saved outputs. The separate tune adapter uses supplied NPZ probabilities/indices and labels/indices with source hashes and explicit opt-in, then writes NEW_REPRODUCTION_THRESHOLDS.json. The selected threshold JSON is never overwritten. Refer to examples/tune_job.json; tune only on the declared tuning partition and disclose selection exposure.

No new model inference, cache, student training or threshold fit was run during this release preparation. The prior 50-forward experimental budget remains exhausted and was not reset.
