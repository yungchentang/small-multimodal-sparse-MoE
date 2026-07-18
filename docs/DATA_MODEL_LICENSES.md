# Data and model provenance

This repository publishes code, compact metrics, and report artifacts. It does
not redistribute source datasets, media, pretrained weights, derived feature
caches, or project checkpoints. The repository MIT license covers only code
and documentation authored for this project; it does not relicense upstream
content.

## Pretrained models

| Role | Hugging Face identifier | Experiment revision | Upstream license note |
|---|---|---|---|
| Sparse language model | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | Apache-2.0 according to the upstream OLMoE release; verify the pinned model card and responsible-use terms. |
| Image encoder | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | The pinned Hugging Face card has no blanket license field. Review the OpenAI CLIP repository/model terms before downloading or redistributing weights. |
| Speech encoder | `openai/whisper-base.en` | `911407f4214e0e1d82085af863093ec0b66f9cd6` | Apache-2.0 in the upstream model card. |
| Optional speech smoke encoder | `openai/whisper-tiny.en` | `87c7102498dcde7456f24cfd30239ca606ed9063` | Apache-2.0 in the upstream model card. This model is not used for final E3 evidence. |

Model pages:

- <https://huggingface.co/allenai/OLMoE-1B-7B-0924>
- <https://huggingface.co/openai/clip-vit-base-patch32>
- <https://huggingface.co/openai/whisper-base.en>
- <https://huggingface.co/openai/whisper-tiny.en>

The revisions above are the commits resolved in the Run:AI cache used for the
reported experiments. Public builders should pass these revisions explicitly;
using a moving `main` branch may not reproduce the checkpoint byte-for-byte.

## Training and evaluation data

| Task | Dataset identifier | Experiment revision | License/terms note |
|---|---|---|---|
| General text | `NeelNanda/c4-10k` | `bdb17e3672308890562fe8f5ebe5d07bc88d764a` | Derived C4 subset; review C4 and Common Crawl terms. |
| General text fallback | `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | ODC-BY 1.0 plus Common Crawl terms of use. |
| Code | `codeparrot/codeparrot-clean-valid` | `4db92d2ec0c1b4c41eeb439cfae16854511d9dcd` | Mixed source-file licenses stored in the row-level `license` field; no blanket relicensing. |
| Logic/reasoning | `hails/agieval-logiqa-en` | `e01aa0f040456cd8d58ee9986a58b64e26c1b782` | Benchmark and underlying exam-material terms must be reviewed. |
| Mathematics | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | MIT in the dataset card. |
| Education | `hails/agieval-sat-en` | `848ee12cf003124f5a1e33446fa3cc6d2ec028e4` | Benchmark and underlying exam-material terms must be reviewed. |
| Education | `hails/agieval-sat-math` | `51ab661f5a2e48370671c87c29d037b5b2b4853e` | Benchmark and underlying exam-material terms must be reviewed. |
| Education | `hails/agieval-lsat-ar` | `052cc636b612f5563329dd182fb6c2cad56681c8` | Benchmark and underlying exam-material terms must be reviewed. |
| Education | `hails/agieval-lsat-lr` | `d876c675a8d47aa4d8a6d682ca8400b7d2ffe1c4` | Benchmark and underlying exam-material terms must be reviewed. |
| Image-caption training/development | `jxie/coco_captions` | `a2ed90d49b61dd13dd71f399c70f5feb897f8bec` | COCO images and annotations retain their original terms; the project does not redistribute them. |
| Sealed image-caption validation | `Multimodal-Fatima/COCO_captions_validation` | `bfa149029bb1e2975cb0b9bea8ad948db9e9ddb2` | COCO validation images and captions retain their original terms; this source is used only for sealed evaluation and is not redistributed. |
| Speech-transcript | `openslr/librispeech_asr` | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | CC BY 4.0 in the dataset card. |

Dataset pages are under `https://huggingface.co/datasets/<identifier>`. The
builders download and preprocess these sources locally. Dataset contents,
packed token blocks, media, transcripts, and sealed rows remain excluded from
Git. Users are responsible for attribution, access, and downstream-use
obligations associated with every source.

## Derived artifacts

Checkpoints and feature caches are not committed because of size and upstream
terms. The public evidence bundle contains configuration, hashes, aggregate
metrics, content-free per-query ranks, report figures, and Run:AI provenance.
A checkpoint SHA-256 identifies the exact local artifact used in the report,
but a hash alone does not permit independent inference or prove model behavior.
