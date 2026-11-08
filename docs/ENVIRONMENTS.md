# Recorded environments

Install PyTorch and its matching CUDA runtime using the official PyTorch instructions for your system. Dependency files do not replace PyTorch. Keep inference and training environments separate.

- The single-round 2026-10-08 T4 deployment comparison recorded torch 2.11.0+cu130 and transformers 5.16.1.
- An earlier PKP2 T4 pass used transformers 5.17.0; `requirements-inference.txt` preserves that version from the original portable source candidate. It is not a newly tested installation.
- Original training: Python 3.11.16, torch 2.10.0+cu128, transformers 4.57.6. `requirements-training.txt` preserves its recorded direct dependencies.
- Weight-edit research used local torch 2.2.2+cu118 and a different Transformers runtime. It does not certify the frozen general CLI on a new Colab instance.

Tests in this repository use only NumPy/stdlib, patch any attempted model import, and require no GPU or network. Successful tests verify contracts, not real inference throughput or fresh environment installation.
