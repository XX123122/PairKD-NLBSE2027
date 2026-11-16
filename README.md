# PairKD: two-layer joint Weakness and SATD classification

PairKD uses one shared two-layer CodeBERT encoder to predict `[Weakness, MAT]` from a source-code `Function`. MAT is the competition's SATD annotation. This repository candidate accompanies the NLBSE 2027 Code Comment Classification tool paper by **Xin Xu and Jun Yu (corresponding author)**, University of Science and Technology of China.

**Selected model: original PKP2-seed13, epoch4.** The experimental weight edits in the paper are diagnostics and are not deployment models. Source repository: https://github.com/XX123122/PairKD-NLBSE2027. The original model and fixed teacher/split prerequisites are distributed through [Releases](https://github.com/XX123122/PairKD-NLBSE2027/releases). See [release status](docs/RELEASE_STATUS.md) for concrete remaining items.

## What is included

| Location | Purpose |
|---|---|
| `src/`, `cli.py`, `bootstrap_assets.py` | Ordered Function-only inference and explicit training/cache/tuning adapters |
| `scripts/import_model_zip.py` | Verify the exact original deployment ZIP and import its checkpoint |
| `scripts/evaluate_saved.py` | Evaluate already saved ordered predictions with separately supplied labels |
| `tests/` | Synthetic, zero-model tests for input contracts, threshold boundaries and saved evaluation |
| `docs/` | Installation, reproduction scope, asset inventory, model card, licenses and submission status |
| `notebooks/PairKD_inference.ipynb` | Local-repository/Colab workflow; real inference defaults to disabled |
| `paper/PairKD.pdf`, `paper/main.tex` | Four-page named tool-paper draft and LaTeX source |
| `results/` | Archived T4 and threshold comparisons, not newly computed scores |

## Install and test

For contract tests only, install `requirements-test.txt`, then run:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

For real inference, install a compatible **PyTorch/CUDA runtime first**, then the recorded inference dependencies. See [environment details](docs/ENVIRONMENTS.md); tested historical versions do not establish compatibility with every new machine.

```bash
python -m pip install -r requirements-inference.txt
python scripts/import_model_zip.py --model-zip /path/to/PairKD-PKP2-seed13-epoch4-v0.1.0.zip --assets-root my_assets
python cli.py verify-assets --assets-root my_assets
python cli.py infer --input examples/functions.jsonl --output predictions.jsonl --assets-root my_assets --dry-run
python cli.py infer --input examples/functions.jsonl --output predictions.jsonl --assets-root my_assets --device cuda:0
```

The first command after installation imports assets without constructing a model. Obtain the exact model ZIP from the [release](https://github.com/XX123122/PairKD-NLBSE2027/releases/tag/v0.2.0). The checkpoint is stored as a Release attachment, separately from ordinary Git history. A pre-extracted original checkpoint can instead be passed to `bootstrap_assets.py --checkpoint ... --assets-root my_assets`.

JSONL rows contain exactly integer `index` and string `Function`. Indices may repeat and order is preserved. Outputs contain `index`, `prob_weakness`, `prob_mat`, `pred_weakness`, `pred_mat`. Extra label/project/filepath fields are rejected; an empty input produces empty output without loading a model. CUDA inference uses fixed batch 600, global width 512, FP16 forward and FP32 sigmoid. OOM propagates; there is no silent microbatch or CPU fallback.

## Model and results

The selected checkpoint SHA256 is `3a902b6b82b48efcd718fabe120ebc1650510c1b4e1efd45625ca3c60cf2b1c4`. It has **53,771,526 parameters**, teacher layers `[0,11]`, hidden width 768 and FFN 3072. Frozen float32 thresholds are `[0.5899999737739563, 0.9700000286102295]`, applied with `>=` in Weakness/MAT order. They were historically fitted on the public validation split, which is development-exposed.

| 2026-10-08 single-pass T4 record | Value |
|---|---:|
| Weakness F1 / MAT F1 | 0.892311 / 0.924528 |
| Mean F1 | 0.908419 |
| Forward-equivalent time | 33.1634 seconds |
| Profiler-reported GFLOPs | 444,468.6284 |
| Executable-reference score | 0.908788 |

The earlier 2026-10-07 ten-pair comparison gives PKP2 mean time 39.87 s and mean reference score .907447; it selected PKP2 over PKP3 in all ten pairs. These are different sessions and must not be pooled or used interchangeably. See [results scope](docs/RESULTS.md). Historical training used RTX 4090; the weight-edit diagnostics used RTX 3050 Laptop GPU. Public deployment measurements are Tesla T4.

## Train, tune and evaluate

The selected recipe uses the original 193,124/21,462 split, Pair+CLS42 teacher, a fresh seed13 head, four epochs and 8,048 updates. The two-layer student learns with 0.5 weighted BCE + 0.5 Bernoulli KL at T=2. Full configuration is [FROZEN_CONFIG.json](FROZEN_CONFIG.json).

```bash
python cli.py cache --job examples/cache_job.json
python cli.py train --job examples/train_job.json
python cli.py tune --job examples/tune_job.json
python scripts/evaluate_saved.py --predictions predictions.jsonl --gold ordered_labels.jsonl --output metrics.json
```

Training/cache/tune commands default to input-validation dry runs. The examples are low-level templates. The preparation and cache-binding scripts in docs/REPRODUCIBILITY.md generate concrete, hash-bound jobs from the official data and released teacher/split prerequisites. Explicit execution also requires `explicit_future_execution_consent: true` and `--execute`. Tuning writes a separate reproduction policy and never changes the selected thresholds. See [reproduction scope](docs/REPRODUCIBILITY.md) for the teacher-based reproduction route and the distinction between recipe reproduction and bit-identical historical checkpoint regeneration. This preparation does not claim a new full training rerun.

## Paper, attribution and release

Read [the paper](paper/PairKD.pdf), [model card](docs/MODEL_CARD.md), [third-party notices](NOTICE.md) and [license preparation](docs/LICENSING.md). Code and project modifications use the MIT license, with upstream notices retained. The manuscript links this repository. Full fresh-environment retraining and organizer-interface acceptance remain unverified. No competition form has been submitted.

Official task: https://github.com/nlbse2027/code-comment-classification

Tool competition: https://nlbse2027.github.io/tools/
