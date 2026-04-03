from __future__ import annotations

import os
import json
from tqdm import trange

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase
from utils.paint import plot_1d_distribution_hist


def compute_activation_value(
    model_base: ModelBase,
    sae_base: SAEBase,
    chunk_samples: int = 2048,
) -> dict:
    """
    Compute per-latent activation value stats (mean/min/max) from cached SAE Top-K (idx, val).

    Requires that you have already run:
        - get_llm_activations(...)  -> writes ./data/{model_name}/ids.dat and meta.json
        - get_sae_activation(...)   -> writes:
              ./data/{model_name}/{sae_name}/layer-{layer}/sae_topk_idx.dat
              ./data/{model_name}/{sae_name}/layer-{layer}/sae_topk_val.dat   (NEW REQUIRED)

    Saves:
        - activation_value_stats.pt  (dict: mean/min/max/count)
        - distribution_mean.png
        - cdf_mean.png

    Returns a dict with output paths.
    """
    model_name = model_base.model_name
    sae_name = sae_base.sae_name
    layer = sae_base.layer
    top_k = sae_base.top_k
    n_feat = sae_base.sae_dim

    llm_dir = os.path.join("./data", model_name, sae_name)
    sae_dir = os.path.join(llm_dir, f"layer-{layer}")

    save_dir = os.path.join(sae_dir, "attribute", "activation_value")
    stats_path = os.path.join(save_dir, "stats.pt")

    if os.path.exists(stats_path):
        saved = torch.load(stats_path, weights_only=False)
        val_avg = saved["avg"].numpy()
        dist_path = os.path.join(save_dir, "distribution_mean.png")
        plot_1d_distribution_hist(
            data=val_avg,
            mask=np.isfinite(val_avg),
            save_path=dist_path,
            title="Distribution of per-latent mean activation value",
            xlabel="Per-latent mean activation value",
            ylabel="Number of SAE latents",
            bins=120,
        )
        return {
            "stats_pt": stats_path,
            "distribution_mean_png": dist_path,
        }

    meta_path = os.path.join(llm_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing meta.json at {meta_path}. Run get_llm_activations first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_sample = int(meta["n_sample"])
    seq_len = int(meta["seq_len"])

    ids_path = os.path.join(llm_dir, "ids.dat")
    idx_path = os.path.join(sae_dir, "sae_topk_idx.dat")
    val_path = os.path.join(sae_dir, "sae_topk_val.dat")

    ids_mm = np.memmap(ids_path, dtype="int32", mode="r", shape=(n_sample, seq_len))
    idx_mm = np.memmap(idx_path, dtype="int32", mode="r", shape=(n_sample, seq_len, top_k))
    val_mm = np.memmap(val_path, dtype="float32", mode="r", shape=(n_sample, seq_len, top_k))

    os.makedirs(save_dir, exist_ok=True)

    bos_id = model_base.tokenizer.bos_token_id
    eos_id = model_base.tokenizer.eos_token_id
    pad_id = model_base.tokenizer.pad_token_id

    val_sum = np.zeros((n_feat,), dtype=np.float32)
    val_min = np.full((n_feat,), np.inf, dtype=np.float32)
    val_max = np.full((n_feat,), -np.inf, dtype=np.float32)
    val_count = np.zeros((n_feat,), dtype=np.int32)

    def _update(feat: int, v: float):
        val_sum[feat] += v
        val_count[feat] += 1
        if v < val_min[feat]:
            val_min[feat] = v
        if v > val_max[feat]:
            val_max[feat] = v

    for start in trange(0, n_sample, chunk_samples, desc="Activation Value"):
        end = min(start + chunk_samples, n_sample)
        B = end - start

        ids_blk = ids_mm[start:end]        # (B, seq_len)
        idx_blk = idx_mm[start:end]        # (B, seq_len, top_k)
        val_blk = val_mm[start:end]        # (B, seq_len, top_k)

        tok_mask = (ids_blk != bos_id) & (ids_blk != eos_id)
        if pad_id is not None:
            tok_mask &= (ids_blk != pad_id)
        pos_mask = tok_mask[:, :, None]
        feat_mask = (idx_blk >= 0) & (idx_blk < n_feat)

        mask = pos_mask & feat_mask
        if not np.any(mask):
            continue

        feats = idx_blk[mask]
        vals  = val_blk[mask]

        val_sum += np.bincount(feats, weights=vals, minlength=n_feat)
        val_count += np.bincount(feats, minlength=n_feat)
        np.minimum.at(val_min, feats, vals)
        np.maximum.at(val_max, feats, vals)
    
    val_avg = val_sum / val_count

    stats_path = os.path.join(save_dir, "stats.pt")
    torch.save(
        {
            "avg": torch.from_numpy(val_avg),
            "min": torch.from_numpy(val_min),
            "max": torch.from_numpy(val_max),
        },
        stats_path,
    )

    dist_path = os.path.join(save_dir, "distribution_mean.png")
    plot_1d_distribution_hist(
        data=val_avg,
        mask=(val_count!=0),
        save_path=dist_path,
        title="Distribution of per-latent mean activation value",
        xlabel="Per-latent mean activation value",
        ylabel="Number of SAE latents",
        bins=120,
    )

    return {
        "stats_pt": stats_path,
        "distribution_mean_png": dist_path,
    }


def run(model_base: ModelBase, sae_base: SAEBase):
    out = compute_activation_value(model_base=model_base, sae_base=sae_base)
    print(f"[activation_value] stats: {out['stats_pt']}")
    print(f"[activation_value] distribution: {out['distribution_mean_png']}")
    return out