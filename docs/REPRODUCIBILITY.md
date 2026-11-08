# Reproduction scope and required inputs

## Reuse the measured checkpoint

`scripts/import_model_zip.py` accepts only the exact frozen deployment ZIP, verifies the checkpoint before import, and prepares the asset layout used by `cli.py`. No training labels or teacher are required for inference. New predictions can be evaluated separately using `scripts/evaluate_saved.py`; gold rows must have exactly `index` and `Label` (two binary values) in identical order to saved predictions.

## Reproduce the student recipe

The source includes cache/train/tune implementations, not a turnkey verified reconstruction of the entire historical warm-start/teacher/student chain. Required inputs for the frozen student adapter:

| Input | Expected identity / scope | Local state at inventory |
|---|---|---|
| Split NPZ | f0e361f9bf741dd8f6c81137e9977d5a28332c0949627f0d407cc067de0eafe9 | Available to author; not bundled |
| Pair+CLS42 teacher | 447baa87fd51a5e3519c6662f60d2a42e5e88aebcf42ac84441559b418110c0f | Available to author; not bundled |
| Original TRAIN logits | 5e953e450ba83e0eeaef805dc72b9bad6857e89fa21c9c085dbd1282432dd3c8 | Standalone local copy not located |
| TRAIN row indices | e6885ee523a1ced0f0b5664dbdd01798da8b67d889dec0650c26d7872352ee3b | Standalone local copy not located |
| Fresh seed13 head template | aabcfbee06cb51959b3b70b6301882c68bd0dcb7a689a506feb1cf71e3db2f73 | Standalone local copy not located |
| Packed TRAIN features | 193,124 rows in original split order, exact IDs/masks/labels | A portable input file still needs preparation |

`examples/*_job.json` declare root-relative paths and hash bindings. The paths are illustrative and must be supplied. No automatic network fetch or remote training occurs. Fresh teacher caches require separately bound `new_recipe_reproduction` lineage; their hash need not equal the original. The adapter uses workers0, unlike the original workers4 execution, and does not promise bit-identical weights or timing. It records epoch2 diagnostically and fixes epoch4 as endpoint.

The earlier raw supervised warm start and Pair teacher training are part of the model lineage. The small source adapter does not independently rebuild those stages. Formal release must either include a complete tested reconstruction pipeline and obtainable inputs or accurately narrow its claimed reproduction scope. Existing model scores must not be silently assigned to newly trained endpoints.

## Evaluation

`scripts/evaluate_saved.py` reports both frozen and fixed-0.5 policies from saved probabilities, using >= for students. It never searches thresholds, runs models or estimates runtime. `cli.py reference-score` computes only the disclosed formula; it is not an official ranking service. Baseline records use their historical >0.5 policy and must not be mixed into the student evaluator as though the decisions were identical at equality.
