from __future__ import annotations

import os
import json
from typing import Optional, Tuple
from tqdm import trange

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase
from utils.paint import plot_binned_proportion_bar

from .utils import decoder_entropy

def compute_entropy(
    model_base: ModelBase,
    sae_base: SAEBase,
    feature_chunk: int = 8,
    min_total_count_for_ratio_plots: int = 1000,
) -> dict:
    """Compute decoder entropy for all SAE features and save artifacts."""
    model_name = model_base.model_name
    sae_name = sae_base.sae_name
    layer = sae_base.layer

    llm_dir = os.path.join("./data", model_name, sae_name)
    sae_dir = os.path.join(llm_dir, f"layer-{layer}")

    save_dir = os.path.join(sae_dir, "attribute", "entropy")
    os.makedirs(save_dir, exist_ok=True)
    entropy_path = os.path.join(save_dir, "entropy.pt")

    if os.path.exists(entropy_path):
        ent_np = torch.load(entropy_path, weights_only=False).numpy()
        bins = np.arange(0.0, 12.1, 1.0)
        fig_path = os.path.join(save_dir, "entropy_bar.png")
        plot_binned_proportion_bar(
            data=ent_np,
            bins=bins,
            save_path=fig_path,
            title="Entropy Distribution",
            xlabel="Entropy Interval",
            ylabel="Proportion of Features",
            descending=True,
            rotate_xticks=45,
        )
        return {
            "save_dir": save_dir,
            "entropy_pt": entropy_path,
            "fig": fig_path,
        }

    # Model components
    norm = model_base._get_model_norm_modules()
    lm_head = model_base._get_model_lm_head().weight

    # Prefer the lm_head device (more stable under device_map="auto").
    device = lm_head.device
    norm = norm.to(device)

    # SAE decoder vectors
    W_dec = sae_base.sae["W_dec"].to(device=device, dtype=torch.float32)  # [n_feat, d_model]
    n_feat = W_dec.shape[0]

    ent = torch.empty((n_feat,), dtype=torch.float32, device=device)

    for start in trange(0, n_feat, int(feature_chunk), desc="Entropy"):
        end = min(start + int(feature_chunk), n_feat)
        vecs = W_dec[start:end, :].contiguous()
        ent[start:end] = decoder_entropy(vecs, norm, lm_head)[0]

    entropy_path = os.path.join(save_dir, "entropy.pt")
    torch.save(ent.detach().cpu(), entropy_path)

    bins = np.arange(0.0, 12.1, 1.0)
    fig_path = os.path.join(save_dir, "entropy_bar.png")
    plot_binned_proportion_bar(
        data=ent.detach().cpu().numpy(),
        bins=bins,
        save_path=fig_path,
        title="Entropy Distribution",
        xlabel="Entropy Interval",
        ylabel="Proportion of Features",
        descending=True,
        rotate_xticks=45,
    )

    out = {
        "save_dir": save_dir,
        "entropy_pt": entropy_path,
        "fig": fig_path,
    }

    return out


def run(
    model_base: ModelBase,
    sae_base: SAEBase,
):
    res = compute_entropy(model_base=model_base, sae_base=sae_base)
    print(f"[entropy] saved: {res.get('entropy_pt')}")
    print(f"[entropy] fig: {res.get('fig')}")
    return res