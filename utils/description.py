from __future__ import annotations

import json
import os
import heapq
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase
from attribute.entropy.utils import decoder_entropy


def _count_all_token_ids(
    ids_mm: np.memmap,
    vocab_size: int,
    chunk_samples: int = 2048,
) -> np.ndarray:
    counts = np.zeros((int(vocab_size),), dtype=np.int64)
    n_sample = ids_mm.shape[0]
    for start in range(0, n_sample, int(chunk_samples)):
        end = min(start + int(chunk_samples), n_sample)
        ids = np.asarray(ids_mm[start:end], dtype=np.int64)
        flat = ids.reshape(-1)
        valid = (flat >= 0) & (flat < int(vocab_size))
        if not np.any(valid):
            continue
        counts += np.bincount(flat[valid], minlength=int(vocab_size))
    return counts


def _make_left_padding_attention_mask(
    prefix_ids: torch.Tensor, pad_id: Optional[int]
) -> torch.Tensor:
    attn = torch.ones_like(prefix_ids, dtype=torch.long)
    if pad_id is None:
        return attn
    ids0 = prefix_ids[0].detach().to("cpu").numpy().tolist()
    pad_len = 0
    for t in ids0:
        if int(t) == int(pad_id):
            pad_len += 1
        else:
            break
    if pad_len > 0:
        attn[0, :pad_len] = 0
    return attn


def describe_features(
    model_base: ModelBase,
    sae_base: SAEBase,
    feature_ids: List[int],
    text_k: int = 40,
    chunk_samples: int = 1024,
    data_root: str = "./data",
) -> Dict[int, str]:
    """Generate description files for selected SAE latents.

    For each feature, produces:
        token.jsonl           - per-token activation counts
        topk_text.jsonl       - top-k texts ranked by mean activation value across the full text
        decoder_entropy.json  - decoder entropy + top-10 predicted tokens
        feature.json          - summary statistics

    Args:
        model_base:    Loaded model wrapper.
        sae_base:      Loaded SAE wrapper.
        feature_ids:   List of latent indices to describe.
        text_k:        Number of top activating texts to include.
        chunk_samples: Chunk size when scanning memmap files.
        data_root:     Root directory containing cached data.

    Returns:
        Dict mapping feature_idx -> output directory path.
    """
    tokenizer = model_base.tokenizer

    llm_dir = os.path.join(data_root, model_base.model_name, sae_base.sae_name)
    meta_path = os.path.join(llm_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_sample = int(meta["n_sample"])
    seq_len = int(meta["seq_len"])
    top_k = int(sae_base.top_k)

    ids_path = os.path.join(llm_dir, "ids.dat")
    sae_dir = os.path.join(llm_dir, f"layer-{sae_base.layer}")
    idx_path = os.path.join(sae_dir, "sae_topk_idx.dat")
    vals_path = os.path.join(sae_dir, "sae_topk_val.dat")

    ids_mm = np.memmap(ids_path, dtype="int32", mode="r", shape=(n_sample, seq_len))
    idx_mm = np.memmap(idx_path, dtype="int32", mode="r", shape=(n_sample, seq_len, top_k))
    vals_mm = np.memmap(vals_path, dtype="float32", mode="r", shape=(n_sample, seq_len, top_k))

    vocab_size = len(tokenizer)
    total_counts = _count_all_token_ids(ids_mm, vocab_size=vocab_size, chunk_samples=chunk_samples)

    device = next(model_base.model.parameters()).device

    # Load precomputed attributes
    ratios = torch.load(
        os.path.join(sae_dir, "attribute", "functional_ratio", "ratios.pt"), map_location="cpu"
    )
    act_length = torch.load(
        os.path.join(sae_dir, "attribute", "activation_length", "stats.pt"), map_location="cpu"
    )
    act_value = torch.load(
        os.path.join(sae_dir, "attribute", "activation_value", "stats.pt"), map_location="cpu"
    )
    output_sens = torch.load(
        os.path.join(sae_dir, "attribute", "output_sensitivity", "sensitivity.pt"), map_location="cpu", weights_only=False
    )

    norm = model_base._get_model_norm_modules()
    lm_head = model_base._get_model_lm_head().weight

    out_root = os.path.join(sae_dir, "description")
    os.makedirs(out_root, exist_ok=True)

    total_positions = n_sample * seq_len
    result: Dict[int, str] = {}

    for feature_idx in feature_ids:
        feature_idx = int(feature_idx)
        if feature_idx < 0 or feature_idx >= int(sae_base.sae_dim):
            print(f"[skip] feature_idx {feature_idx} out of range (0..{sae_base.sae_dim - 1})")
            continue

        feature_dir = os.path.join(out_root, f"feature_{feature_idx}")
        os.makedirs(feature_dir, exist_ok=True)

        ratio_value = ratios[feature_idx].item()
        avg_length = act_length["avg"][feature_idx].item()
        max_length = act_length["max"][feature_idx].item()
        min_length = act_length["min"][feature_idx].item()
        avg_value = act_value["avg"][feature_idx].item()
        max_value = act_value["max"][feature_idx].item()
        min_value = act_value["min"][feature_idx].item()
        mean_sens = output_sens["mean_abs_grad"][feature_idx].item()
        max_sens = output_sens["max_abs_grad"][feature_idx].item()

        # ---- token.jsonl ----
        token_out_path = os.path.join(feature_dir, "token.jsonl")
        with open(token_out_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"feature_idx": feature_idx, "ratio": ratio_value}, ensure_ascii=False)
                + "\n"
            )

        full_vocab_size = len(tokenizer)
        act_counts = np.zeros((full_vocab_size,), dtype=np.int64)
        active_total = 0
        heap: List[Tuple[float, int]] = []  # (avg_activation, sample_idx)

        for s0 in range(0, n_sample, chunk_samples):
            s1 = min(s0 + chunk_samples, n_sample)
            ids_chunk = np.asarray(ids_mm[s0:s1], dtype=np.int64)
            idx_chunk = np.asarray(idx_mm[s0:s1], dtype=np.int64)
            vals_chunk = np.asarray(vals_mm[s0:s1], dtype=np.float32)

            mask = idx_chunk == feature_idx  # (chunk, seq_len, top_k)
            any_mask_pos = mask.any(axis=-1)   # (chunk, seq_len)
            any_mask_sample = any_mask_pos.any(axis=-1)  # (chunk,)

            if not np.any(any_mask_sample):
                continue

            active_total += int(any_mask_pos.sum())

            tok_ids = ids_chunk[any_mask_pos].reshape(-1)
            valid = (tok_ids >= 0) & (tok_ids < full_vocab_size)
            if np.any(valid):
                act_counts += np.bincount(tok_ids[valid], minlength=full_vocab_size)

            # Per-sample average activation (non-activated positions count as 0)
            feat_vals_pos = (vals_chunk * mask).sum(axis=-1)  # (chunk, seq_len)
            sample_avg = feat_vals_pos.mean(axis=-1)  # (chunk,)

            for local_idx in np.flatnonzero(any_mask_sample).tolist():
                global_idx = s0 + int(local_idx)
                avg = float(sample_avg[local_idx])
                if len(heap) < text_k:
                    heapq.heappush(heap, (avg, global_idx))
                elif avg > heap[0][0]:
                    heapq.heapreplace(heap, (avg, global_idx))

        with open(token_out_path, "a", encoding="utf-8") as f:
            f.write(f"activate / total: {active_total} / {total_positions}\n")
            entries = []
            for token_id in np.nonzero(act_counts)[0].tolist():
                tok = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
                entries.append(
                    {
                        "token": tok,
                        "activate_num": int(act_counts[token_id]),
                        "total_num": int(total_counts[token_id]),
                    }
                )
            entries.sort(key=lambda x: x["activate_num"], reverse=True)
            for row in entries:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # ---- topk_text.jsonl ----
        topk_out_path = os.path.join(feature_dir, "topk_text.jsonl")
        with open(topk_out_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"feature_idx": feature_idx, "ratio": ratio_value}, ensure_ascii=False)
                + "\n"
            )

        heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)

        with open(topk_out_path, "a", encoding="utf-8") as f:
            for avg_act, sample_idx in heap_sorted:
                full_ids = np.asarray(ids_mm[sample_idx, :], dtype=np.int64)
                full_idx = np.asarray(idx_mm[sample_idx, :, :], dtype=np.int64)

                tokens = tokenizer.convert_ids_to_tokens(full_ids.tolist())
                for t in range(seq_len):
                    if np.any(full_idx[t] == feature_idx):
                        tokens[t] = f"[{tokens[t]}]"

                text = tokenizer.convert_tokens_to_string(tokens)
                entry = {
                    "avg_activation": float(avg_act),
                    "text": text,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # ---- decoder_entropy.json ----
        decoder_vec = sae_base.sae["W_dec"][feature_idx : feature_idx + 1, :]
        H, topk_ids, topk_probs = decoder_entropy(
            decoder_vec=decoder_vec,
            norm=norm.to(device),
            lm_head=lm_head.to(device),
            topk=10,
        )
        topk_ids_list = topk_ids.flatten().tolist()
        topk_probs_list = topk_probs.flatten().tolist()
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids_list)
        with open(os.path.join(feature_dir, "decoder_entropy.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_idx": feature_idx,
                    "decoder_entropy": float(H),
                    "top10_decoder_tokens": [
                        {"token": tok, "token_id": int(tid), "prob": float(p)}
                        for tok, tid, p in zip(topk_tokens, topk_ids_list, topk_probs_list)
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ---- feature.json ----
        activate_ratio = float(active_total / total_positions)
        with open(os.path.join(feature_dir, "feature.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "activate_ratio": f"{activate_ratio:.6f} ({active_total} / {total_positions})",
                    "functional ratio": f"{ratio_value:.4f}",
                    "entropy": f"{H:.2f}",
                    "activation length(avg)": f"{avg_length:.2f}",
                    "activation length(min)": f"{min_length:.2f}",
                    "activation length(max)": f"{max_length:.2f}",
                    "activation value(avg)": f"{avg_value:.2f}",
                    "activation value(min)": f"{min_value:.2f}",
                    "activation value(max)": f"{max_value:.2f}",
                    "output sensitivity(mean)": f"{mean_sens:.6f}",
                    "output sensitivity(max)": f"{max_sens:.6f}",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"latent {feature_idx:5d}: activate ratio {activate_ratio:.6f}, "
            f"functional ratio {ratio_value:.4f}, entropy {H:.2f}, "
            f"activation length(avg) {avg_length:.2f}, "
            f"activation value(avg) {avg_value:.2f}, "
            f"output sensitivity(mean) {mean_sens:.6f}"
        )
        result[feature_idx] = feature_dir

    return result
