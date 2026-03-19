from __future__ import annotations

import os
import json
import shutil
from typing import Optional
from tqdm import trange

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase
from utils.paint import plot_1d_distribution_hist

def compute_activation_length(
    model_base: ModelBase,
    sae_base: SAEBase,
    chunk_samples: int = 2048,
) -> dict:
    """
    Compute activation_length for all SAE features on the cached SAE Top-K indices.

    Requires that you have already run:
        - get_llm_activations(...)  -> writes ./data/{model_name}/ids.dat and meta.json
        - get_sae_activation(...)   -> writes ./data/{model_name}/{sae_name}/layer-{layer}/sae_topk_idx.dat

    Returns a dict with output paths.
    """
    model_name = model_base.model_name
    sae_name = sae_base.sae_name
    layer = sae_base.layer
    top_k = sae_base.top_k
    n_feat = sae_base.sae_dim

    llm_dir = os.path.join("./data", model_name)
    sae_dir = os.path.join(llm_dir, sae_name, f"layer-{layer}")

    meta_path = os.path.join(llm_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json at {meta_path}. Run get_llm_activations first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_sample = int(meta["n_sample"])
    seq_len = int(meta["seq_len"])

    ids_path = os.path.join(llm_dir, "ids.dat")
    idx_path = os.path.join(sae_dir, "sae_topk_idx.dat")

    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Missing ids.dat at {ids_path}.")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing sae_topk_idx.dat at {idx_path}. Run get_sae_activation first.")

    ids_mm = np.memmap(ids_path, dtype="int32", mode="r", shape=(n_sample, seq_len))
    idx_mm = np.memmap(idx_path, dtype="int32", mode="r", shape=(n_sample, seq_len, top_k))

    save_dir = os.path.join(sae_dir, "attribute", "activation_length")
    os.makedirs(save_dir, exist_ok=True)

    bos_id = model_base.tokenizer.bos_token_id
    eos_id = model_base.tokenizer.eos_token_id
    pad_id = model_base.tokenizer.pad_token_id

    len_sum   = np.zeros((n_feat,), dtype=np.float32)
    len_min   = np.full((n_feat,), np.inf, dtype=np.float32)
    len_max   = np.zeros((n_feat,), dtype=np.float32)
    len_count = np.zeros((n_feat,), dtype=np.int32)

    def _finalize_run(feat: int, ln: int):
        len_sum[feat] += ln
        len_count[feat] += 1
        if ln < len_min[feat]:
            len_min[feat] = ln
        if ln > len_max[feat]:
            len_max[feat] = ln

    for s in trange(n_sample, desc="Activation Length"):
        ids_row = ids_mm[s]   # (seq_len,)
        idx_row = idx_mm[s]   # (seq_len, top_k)

        active_runs = {} 

        for t in range(seq_len):
            tok = int(ids_row[t])

            feats = idx_row[t]  # (top_k,)
            feats = feats[(feats >= 0) & (feats < n_feat)]

            if (tok == bos_id) or (tok == eos_id) or (pad_id is not None and tok == pad_id) or feats.size == 0:
                if active_runs:
                    for feat, ln in active_runs.items():
                        _finalize_run(feat, ln)
                    active_runs.clear()
                continue

            current_set = set(map(int, np.unique(feats).tolist()))

            if active_runs:
                to_finalize = [f for f in active_runs.keys() if f not in current_set]
                for f in to_finalize:
                    _finalize_run(f, active_runs[f])
                    del active_runs[f]

            for f in current_set:
                active_runs[f] = active_runs.get(f, 0) + 1

        if active_runs:
            for feat, ln in active_runs.items():
                _finalize_run(feat, ln)

    len_avg = len_sum / len_count

    stats_path = os.path.join(save_dir, "stats.pt")
    torch.save(
        {
            "avg": torch.from_numpy(len_avg),
            "min":  torch.from_numpy(len_min),
            "max":  torch.from_numpy(len_max),
        },
        stats_path,
    )

    dist_path = os.path.join(save_dir, "distribution_mean.png")
    plot_1d_distribution_hist(
        data=len_avg,
        mask=(len_count!=0),
        save_path=dist_path,
        title="Distribution of per-latent mean activation length",
        xlabel="Per-latent mean activation length",
        ylabel="Number of SAE latents",
        bins=120,
    )

    return {
        "status_pt": stats_path,
        "distribution_mean_png": dist_path,
    }

def run(
    model_base: ModelBase,
    sae_base: SAEBase,
):
    out = compute_activation_length(
        model_base=model_base,
        sae_base=sae_base,
    )
    print(f"[activation_length] status: {out['status_pt']}")
    print(f"[activation_length] distribution: {out['distribution_mean_png']}")
    return out
