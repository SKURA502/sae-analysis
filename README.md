# Fantastic SAE Analysis

A toolkit for systematically understanding the concepts encoded in Sparse Autoencoders (SAEs). Each SAE feature (latent) is characterized from **six quantitative angles**, and then examined from both **input** and **output** perspectives to reveal what concept it represents.

---

## Overview

SAEs decompose LLM residual stream activations into sparse, interpretable features. But understanding what each feature actually *means* is non-trivial. This project provides a structured pipeline to:

1. **Profile every feature** across six statistical attributes (using `main.py`)
2. **Describe individual features** in human-readable detail (using `utils/description.py`)

The six attributes give a global picture of the entire SAE. The description step zooms in on specific features and explains them from both what tokens *trigger* the feature (input perspective) and what tokens the feature *predicts* (output perspective).

---

## Six Feature Attributes

### Input Perspective

| Attribute | What it measures |
|---|---|
| **activate\_ratio** | Fraction of all token positions where this feature fires — reflects how common or rare the concept is in the corpus |
| **functional\_ratio** | Fraction of activations on functional tokens (punctuation, special tokens, etc.) — distinguishes syntactic from semantic features |
| **activation\_length** | Number of consecutive token positions a feature stays active — captures whether a feature is position-local or span-level |
| **activation\_value** | Magnitude of the feature's activation coefficient when it fires |

### Output Perspective

| Attribute | What it measures |
|---|---|
| **entropy** | Shannon entropy of the probability distribution induced by the decoder vector through the LM head — low entropy means the feature strongly predicts specific tokens |
| **output\_sensitivity** | Mean absolute gradient of next-token loss w.r.t. the feature's pre-activation — measures how much model output depends on this feature |

---

## Input & Output Perspectives

After running `main.py`, you can call `utils/description.py` to generate per-feature description files:

| File | Perspective | Content |
|---|---|---|
| `decoder_entropy.json` | **Output** | Decoder entropy score + top-10 tokens predicted by the feature's decoder vector through the LM head |
| `token.jsonl` | **Input** | Per-token activation counts — which vocabulary items most frequently trigger this feature |
| `topk_text.jsonl` | **Input** | Top-K text passages ranked by mean activation value — the contexts where this feature fires most strongly |
| `feature.json` | **Summary** | Compact snapshot of all six attribute values for the feature |

Together, the input and output perspectives let you triangulate a feature's meaning: if a feature fires on tokens like `"rou"`, `"arou"`, `"eye"` and its decoder predicts tokens related to directions or vision, you can confidently label the concept it encodes.

---

## Prerequisites

### Datasets

Download the following datasets to local storage before running:

- **[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** (`sample-10BT` split) — general web text corpus
- **[BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)** — safety-relevant instruction-response pairs
- **[When2Call](https://huggingface.co/datasets/wh1isper/when2call)** — tool-use preference dataset (alternative to BeaverTails for tool-call SAEs)

By default the pipeline mixes FineWeb (50%) with BeaverTails (50%). Pass `--second_dataset when2call` to use When2Call instead.

Update the dataset paths at the top of `dataset/load.py` to match your local setup.

### Model Weights

Download the target LLM and its corresponding SAE checkpoint locally. For a quick start, use the pre-trained **[Llama-Scope](https://huggingface.co/collections/fnlp/llama-scope-6720c1f8373805041022e1e6)** SAEs, which are already supported out of the box.

### Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Run `main.py` to compute feature attributes

```bash
export PYTHONPATH=/data/sae-analysis:$PYTHONPATH

python -u main.py \
    --model_alias meta-llama/Llama-3.1-8B \
    --model_path /path/to/Llama-3.1-8B \
    --sae_name Llama-Scope \
    --sae_path /path/to/Llama-Scope/layer-30/final.safetensors \
    --top_k 50 \
    --layer 30 \
    --dtype bfloat16 \
    --norm_factor 53.25 \
    --device cuda:0
```

Key arguments:

| Argument | Description |
|---|---|
| `--model_alias` | Canonical model identifier (used for output directory naming) |
| `--model_path` | Local path to model weights |
| `--sae_name` | SAE name tag (e.g. `Llama-Scope`, `Llama-Mixed`) |
| `--sae_path` | Local path to the SAE checkpoint |
| `--layer` | Transformer layer the SAE is attached to |
| `--top_k` | SAE top-K sparsity (must match the trained SAE) |
| `--norm_factor` | Activation norm scaling factor (defaults to `sqrt(d_model)` if omitted) |
| `--second_dataset` | `beavertails` (default) or `when2call` |

This produces, under `./data/{model_alias}/{sae_name}/layer-{layer}/attribute/`:

```
attribute/
  functional_ratio/   ratios.pt, counts.pt, ratio_bar.png, counts_rate_hist.png
  entropy/            entropy.pt, entropy_bar.png
  activation_length/  stats.pt, distribution_mean.png
  activation_value/   stats.pt, distribution_mean.png
  output_sensitivity/ sensitivity.pt, distribution_mean_abs_grad.png
  correlation/        pearson_correlation.csv, spearman_correlation.csv, heatmaps
```

See `main.sh` for ready-to-use command examples for different model/SAE combinations.

### Step 2 — Run `utils/description.py` to describe specific features

```python
from utils.description import describe_features

paths = describe_features(
    model_base=model_base,
    sae_base=sae_base,
    feature_ids=[644, 1024, 2048],   # features you want to inspect
    text_k=40,                        # number of top activating texts
)
```

Output for each feature is written to `./data/{model}/{sae}/layer-{layer}/description/feature_{idx}/`:

```
feature_644/
  feature.json          # six-attribute summary
  token.jsonl           # input-perspective: per-token activation stats
  topk_text.jsonl       # input-perspective: top-40 activating texts
  decoder_entropy.json  # output-perspective: entropy + top-10 predicted tokens
```

---

## Project Structure

```
sae-analysis/
  main.py                          # Entry point: compute all six attributes
  main.sh                          # Example launch commands
  requirements.txt
  attribute/
    functional_ratio/              # Functional token ratio computation
    entropy/                       # Decoder entropy computation
    activation_length/             # Activation span length computation
    activation_value/              # Activation magnitude statistics
    output_sensitivity/            # Loss gradient sensitivity computation
  dataset/
    load.py                        # FineWeb / BeaverTails / When2Call loaders
  utils/
    description.py                 # Per-feature description generator
    activation.py                  # LLM + SAE activation caching
    hf_models/                     # Model wrappers (LLaMA, Gemma, Qwen)
    sae/sae_base.py                # SAE checkpoint loader
    paint.py                       # Plotting utilities
```

---

## Quick Start with Llama-Scope

Llama-Scope provides pre-trained SAEs for Llama-3.1-8B across all layers. To try the pipeline immediately:

1. Download `meta-llama/Llama-3.1-8B` weights locally
2. Download the Llama-Scope checkpoint for your target layer (e.g. `Llama3_1-8B-Base-L30R-8x`)
3. Run `main.py` with `--sae_name Llama-Scope --layer 30 --top_k 50`
4. Call `describe_features(...)` on any feature index of interest
