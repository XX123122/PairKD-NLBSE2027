# Licensing preparation

The source tree is an unpublished local candidate. No blanket license grant is applied by this organization task. `LICENSE.preview.MIT` is a concrete proposed MIT license for author-owned project software, to finalize with the authors before release. Prior source attribution and unresolved copied-helper ownership are preserved in `provenance/source_rights_20261007.json`; do not treat this file as a declaration of rights over third-party material.

Upstream Microsoft CodeBERT repository publishes an MIT software license: https://github.com/microsoft/CodeBERT/blob/master/LICENSE . Its model card says CodeBERT is initialized from RoBERTa and trained on CodeSearchNet: https://huggingface.co/microsoft/codebert-base . These references do not alone settle the exact frozen tokenizer/weight/dataset redistribution terms. Preserve upstream notices and check the exact asset origin before public release.

No dataset examples, raw Function fixtures, labels or probability matrices from the competition corpus are included in the public-facing test tree. The two example functions are synthetic. Full upstream baseline code is not copied. Model ZIP rights and its nested historical handoff contents require review as a separate attachment; do not publish the private outer handoff ZIP wholesale.
