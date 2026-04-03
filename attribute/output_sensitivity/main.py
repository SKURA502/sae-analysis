from __future__ import annotations

import os
import json
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import trange
from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase
from utils.paint import plot_1d_distribution_hist


def compute_output_sensitivity(
    model_base: ModelBase,
    sae_base: SAEBase,
    chunk_samples: int = 32,
    clip_percentile: Optional[Tuple[float, float]] = (1.0, 99.0),
) -> dict:
    """Compute per-latent sensitivity of model output w.r.t. SAE latent pre-activations.

    The score is computed as the mean absolute gradient magnitude of the loss (next-token
    cross-entropy) with respect to each SAE latent pre-activation, aggregated over the dataset.

    This is a proxy for how sensitive the model's outputs are to each SAE latent.
    """

    model_name = model_base.model_name
    sae_name = sae_base.sae_name
    layer = sae_base.layer

    llm_dir = os.path.join("./data", model_name, sae_name)
    sae_dir = os.path.join(llm_dir, f"layer-{layer}")

    save_dir = os.path.join(sae_dir, "attribute", "output_sensitivity")
    stats_path = os.path.join(save_dir, "sensitivity.pt")

    if os.path.exists(stats_path):
        saved = torch.load(stats_path, weights_only=False)
        mean_abs_grad = saved["mean_abs_grad"]
        out_plot = os.path.join(save_dir, "distribution_mean_abs_grad.png")
        plot_1d_distribution_hist(
            data=mean_abs_grad,
            save_path=out_plot,
            title="SAE Latent Sensitivity (mean abs grad)",
            xlabel="Mean abs grad",
            ylabel="Number of SAE latents",
            clip_percentile=clip_percentile,
        )
        return {"stats_pt": stats_path, "distribution_mean_abs_grad_png": out_plot}

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
    ids_mm = np.memmap(ids_path, dtype="int32", mode="r", shape=(n_sample, seq_len))

    top_k = int(sae_base.top_k)
    idx_path = os.path.join(sae_dir, "sae_topk_idx.dat")
    val_path = os.path.join(sae_dir, "sae_topk_val.dat")

    if not os.path.exists(idx_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Missing SAE topk cache under {sae_dir}. Run get_sae_activation first."
        )

    idx_mm = np.memmap(idx_path, dtype="int32", mode="r", shape=(n_sample, seq_len, top_k))
    val_mm = np.memmap(val_path, dtype="float32", mode="r", shape=(n_sample, seq_len, top_k))

    os.makedirs(save_dir, exist_ok=True)

    device = model_base.model.device
    pad_id = model_base.tokenizer.pad_token_id
    ignore_index = pad_id if pad_id is not None else -100

    # Hook to capture input to the target layer (residual stream)
    residual_in: dict[str, torch.Tensor] = {}

    def pre_hook_resid(module, inputs):
        x = inputs[0]
        if not x.requires_grad:
            # Model weights are frozen; inject a leaf tensor that requires grad
            # so gradients can flow from the loss back to this point.
            x = x.detach().requires_grad_(True)
            residual_in["value"] = x
            return (x,) + inputs[1:]
        # retain grad on non-leaf tensor so .grad is populated after backward
        x.retain_grad()
        residual_in["value"] = x

    hook_handle = model_base.model_block_modules[int(layer)].register_forward_pre_hook(pre_hook_resid)

    # We'll aggregate stats over all latents
    n_feat = sae_base.sae_dim
    total_abs_grad = np.zeros((n_feat,), dtype=np.float64)
    max_abs_grad = np.zeros((n_feat,), dtype=np.float64)
    count_active = np.zeros((n_feat,), dtype=np.int64)

    # SAE encoding weights
    W_enc = sae_base.sae["W_enc"].to(device=device)
    b_enc = sae_base.sae["b_enc"].to(device=device) if sae_base.sae["b_enc"] is not None else None

    # Iterate over dataset by chunks
    for start in trange(0, n_sample, int(chunk_samples)):
        end = min(start + int(chunk_samples), n_sample)
        B = end - start

        ids_block = torch.from_numpy(np.asarray(ids_mm[start:end], dtype=np.int64)).to(device=device)
        if pad_id is not None:
            attention_mask = (ids_block != pad_id).to(device=device)
        else:
            attention_mask = torch.ones_like(ids_block)

        # Compute next-token loss (autoregressive)
        input_ids = ids_block
        target_ids = ids_block[:, 1:]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        model_base.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L-1, V)

            # Flatten for cross_entropy
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )
            loss.backward()

        # Compute gradients w.r.t SAE latent pre-activations
        grad_resid = residual_in["value"].grad  # (B, L-1, d_model)
        if grad_resid is None:
            raise RuntimeError("Expected residual input grad to be populated. Did you run with torch.enable_grad()?")

        # pre-activation for SAE (before ReLU)
        # W_enc/b_enc may be on a different device than residual when model is sharded
        resid_device = residual_in["value"].device
        W_enc_local = W_enc.to(resid_device)
        preact = torch.einsum("bld,dh->blh", residual_in["value"], W_enc_local)
        if b_enc is not None:
            preact = preact + b_enc.to(resid_device)  # (B, L-1, H)
        active_mask = preact > 0

        grad_latent = torch.einsum("bld,dh->blh", grad_resid, W_enc_local)  # (B, L-1, H)
        abs_grad = grad_latent.abs().detach().cpu().float().numpy()
        active_mask_np = active_mask.detach().cpu().numpy()

        # accumulate
        active_abs_grad = abs_grad * active_mask_np
        total_abs_grad += active_abs_grad.sum(axis=(0, 1))
        max_abs_grad = np.maximum(max_abs_grad, active_abs_grad.max(axis=(0, 1)))
        count_active += active_mask_np.sum(axis=(0, 1))

        # free GPU tensors explicitly before next iteration
        residual_in["value"].grad = None
        residual_in.clear()
        del outputs, logits, loss, grad_resid, preact, active_mask, grad_latent
        torch.cuda.empty_cache()

    hook_handle.remove()

    mean_abs_grad = total_abs_grad / np.maximum(count_active, 1)

    stats = {
        "mean_abs_grad": mean_abs_grad.astype(np.float32),
        "max_abs_grad": max_abs_grad.astype(np.float32),
        "count_active": count_active,
    }

    stats_path = os.path.join(save_dir, "sensitivity.pt")
    torch.save(stats, stats_path)

    out_plot = os.path.join(save_dir, "distribution_mean_abs_grad.png")
    plot_1d_distribution_hist(
        data=mean_abs_grad,
        save_path=out_plot,
        title="SAE Latent Sensitivity (mean abs grad)",
        xlabel="Mean abs grad",
        ylabel="Number of SAE latents",
        clip_percentile=clip_percentile,
    )

    return {"stats_pt": stats_path, "distribution_mean_abs_grad_png": out_plot}


def run(model_base: ModelBase, sae_base: SAEBase, chunk_samples: int = 32):
    out = compute_output_sensitivity(model_base=model_base, sae_base=sae_base, chunk_samples=chunk_samples)
    print(f"[output_sensitivity] stats: {out['stats_pt']}")
    print(f"[output_sensitivity] distribution: {out['distribution_mean_abs_grad_png']}")
    return out
