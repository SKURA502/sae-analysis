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
from utils.paint import plot_binned_proportion_bar

from .functional_token import _load_token_flag_lookup, _ensure_token_flag_map

def compute_functional_ratio(
    model_base: ModelBase,
    sae_base: SAEBase,
    token_coverage: float = 0.40,
    chunk_samples: int = 2048,
) -> dict:
    """
    Compute functional ratio for all SAE features on the cached SAE Top-K indices.

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

    save_dir = os.path.join(sae_dir, "attribute", "functional_ratio")
    os.makedirs(save_dir, exist_ok=True)

    # Ensure token_flag_map.json exists in save_dir (generate if needed)
    token_flag_map_path = _ensure_token_flag_map(
        model_base=model_base,
        ids_mm=ids_mm,
        save_dir=save_dir,
        coverage=float(token_coverage),
    )

    # Token flags (dense lookup)
    lookup = _load_token_flag_lookup(model_base.tokenizer, token_flag_map_path)

    bos_id = model_base.tokenizer.bos_token_id
    eos_id = model_base.tokenizer.eos_token_id
    pad_id = model_base.tokenizer.pad_token_id

    total_count = np.zeros((n_feat,), dtype=np.int64)
    flag1_count = np.zeros((n_feat,), dtype=np.int64)

    # Chunk over samples to control memory
    for start in trange(0, n_sample, chunk_samples, desc="Functional Ratio"):
        end = min(start + chunk_samples, n_sample)

        ids = np.asarray(ids_mm[start:end])  # (B, L)
        idx = np.asarray(idx_mm[start:end])  # (B, L, K)

        # Valid tokens (exclude special)
        valid_tok = np.ones_like(ids, dtype=bool)
        if bos_id is not None:
            valid_tok &= (ids != int(bos_id))
        if eos_id is not None:
            valid_tok &= (ids != int(eos_id))
        if pad_id is not None:
            valid_tok &= (ids != int(pad_id))

        # token flags (0/1), unknown ids default to 0
        ids_clip = ids.copy()
        ids_clip[ids_clip < 0] = 0
        ids_clip[ids_clip >= lookup.shape[0]] = 0
        flags = lookup[ids_clip]  # (B, L) uint8
        flags = (flags == 1) & valid_tok

        # feature validity mask (idx >= 0) and token validity
        valid_feat = (idx >= 0) & valid_tok[..., None]

        # total counts
        if valid_feat.any():
            feat_ids = idx[valid_feat].astype(np.int64, copy=False)
            total_count += np.bincount(feat_ids, minlength=n_feat)

        # flag==1 counts
        flag_mask = valid_feat & flags[..., None]
        if flag_mask.any():
            feat_ids_f = idx[flag_mask].astype(np.int64, copy=False)
            flag1_count += np.bincount(feat_ids_f, minlength=n_feat)

    # Ratios with safe divide
    ratios = np.zeros((n_feat,), dtype=np.float64)
    nonzero = total_count > 0
    ratios[nonzero] = flag1_count[nonzero] / total_count[nonzero]

    counts_pt_path = os.path.join(save_dir, "counts.pt")
    ratios_pt_path = os.path.join(save_dir, "ratios.pt")

    torch.save(torch.from_numpy(total_count), counts_pt_path)
    torch.save(torch.from_numpy(ratios), ratios_pt_path)

    bins = np.arange(0.0, 1.1, 0.1)
    fig_path = os.path.join(save_dir, "ratio_bar.png")
    plot_binned_proportion_bar(
        data=ratios,
        mask=nonzero,
        bins=bins,
        save_path=fig_path,
        title="Functional Ratio Distribution",
        xlabel="Ratio Interval",
        ylabel="Proportion of Features",
        descending=True,
        rotate_xticks=45,
   )

    return {
        "save_dir": save_dir,
        "counts_pt": counts_pt_path,
        "ratios_pt": ratios_pt_path,
        "figure": fig_path,
    }


def run(
    model_base: ModelBase,
    sae_base: SAEBase,
    token_coverage: float = 0.40,
):
    out = compute_functional_ratio(
        model_base=model_base,
        sae_base=sae_base,
        token_coverage=float(token_coverage),
    )
    print(f"[functional_ratio] counts: {out['counts_pt']}")
    print(f"[functional_ratio] ratios: {out['ratios_pt']}")
    print(f"[functional_ratio] fig:    {out['figure']}")
    return out
