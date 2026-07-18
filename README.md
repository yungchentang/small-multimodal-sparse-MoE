# Small Multimodal Sparse MoE

Code for converting the text-only
[`allenai/OLMoE-1B-7B-0924`](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
into a multimodal model while reducing routing from 8 to 2 active experts per
token.

The model uses:

- CLIP ViT-B/32 for images;
- Whisper base.en for speech;
- learned image and audio prefix projectors;
- shared OLMoE prefix-to-text processing;
- an 8-active-expert teacher and 2-active-expert student;
- capacity-aware routing, load balancing, text replay, and selected-expert
  adaptation.

This repository contains source code only. Reports, LaTeX, checkpoints,
datasets, cluster logs, and private experiment artifacts are not included.

## Layout

```text
datasets/     Real-data download and preprocessing
model/        Encoders, fusion, sparse routing, and OLMoE wrapper
training/     Alignment, distillation, and multimodal training
evaluation/   Text and routing metrics
scripts/      Run:AI launchers, frozen evaluation, and analysis
tests/        Unit and integrity tests
notebooks/    Minimal inference walkthrough
```

## Setup

Python 3.10+ and an NVIDIA GPU are recommended. Formal runs use one 80 GB A100.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model and dataset revisions are pinned in [`hf_sources.py`](hf_sources.py).

## Quick Check

The default command runs a small synthetic smoke test. It verifies the code
path only and is not experimental evidence.

```bash
bash run.sh
bash scripts/test_release.sh
```

## Real Data

The data builder downloads public datasets and materializes:

- C4 text;
- CodeParrot code;
- AGIEval logic and educational tasks;
- GSM8K mathematics;
- COCO image-caption pairs;
- LibriSpeech audio-transcript pairs.

```bash
DATA_DIR=data/real_subset \
bash run.sh real-data-prepare

DATA_DIR=data/real_subset \
bash run.sh data-quality-audit
```

Preprocessing includes OLMoE tokenization and 512-token packing, RGB
conversion, CLIP resize/normalization, mono audio loading, 16 kHz resampling,
6-second padding/truncation, batching, prefix concatenation, and prefix label
masking.

Raw datasets are not redistributed. See
[`docs/DATA_MODEL_LICENSES.md`](docs/DATA_MODEL_LICENSES.md).

## Run:AI

Copy the cluster template and fill in the PVC, image, project, and repository
paths:

```bash
cp .env.example .env.runai
$EDITOR .env.runai

MODE=smoke JOB_NAME=sparse-moe-smoke \
bash scripts/submit_runai.sh
```

Each formal job requests at most one GPU. Launchers fail closed when required
paths, hashes, or frozen protocol fields are missing.

## Training

The reproducible training sequence is:

1. modality alignment;
2. 8-to-2 active-expert distillation;
3. train/development-only expert selection;
4. selected-expert multimodal continuation.

```bash
bash scripts/submit_development_alignment_campaign.sh
bash scripts/submit_top2_distill_runai.sh
python scripts/build_train_only_esft_selection.py --help
bash scripts/submit_development_selected_expert_campaign.sh
```

All formal runs use seed 42. Final inference keeps 2 active experts per token.
The exact data, output, checkpoint, and split paths are explicit environment
variables in each launcher.

## Evaluation

Freeze the checkpoint, data split, and evaluator contract before reading final
metrics:

```bash
bash scripts/freeze_final_protocol.sh
bash scripts/submit_sealed_control_matrix.sh
bash scripts/submit_sealed_analysis.sh
```

The matrix evaluates real, shuffled, zero, norm-matched random, and no-prefix
controls for image and speech matching. Additional entry points include:

```bash
bash scripts/submit_whisper_native_wer.sh
bash scripts/submit_representation_funnel.sh
bash scripts/submit_final_specialization.sh
```

Generated data, outputs, checkpoints, and logs are ignored by Git.

## License

Code is released under the [Apache-2.0 license](LICENSE). Model and dataset
licenses remain with their original providers.
