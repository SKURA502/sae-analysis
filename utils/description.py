from __future__ import annotations

import json
import os
import re
import heapq
from typing import Dict, List, Optional, Tuple

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
    context_window_forward: int = 20,
    context_window_backward: int = 20,
    top_grad_k: int = 5,
    chunk_samples: int = 1024,
    data_root: str = "./data",
) -> Dict[int, str]:
    """Generate description files for selected SAE latents.

    For each feature, produces:
        token.jsonl           - per-token activation counts
        topk_text.jsonl       - top-k highest-activating text contexts with gradient highlights
        decoder_entropy.json  - decoder entropy + top-10 predicted tokens
        feature.json          - summary statistics

    Args:
        model_base:               Loaded model wrapper.
        sae_base:                 Loaded SAE wrapper.
        feature_ids:              List of latent indices to describe.
        text_k:                   Number of top activating positions to include in context.
        context_window_forward:   Tokens before the activating position to show.
        context_window_backward:  Tokens after the activating position to show.
        top_grad_k:               Number of top-gradient tokens to highlight.
        chunk_samples:            Chunk size when scanning memmap files.
        data_root:                Root directory containing cached data.

    Returns:
        Dict mapping feature_idx -> output directory path.
    """
    tokenizer = model_base.tokenizer

    llm_dir = os.path.join(data_root, model_base.model_name)
    meta_path = os.path.join(llm_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_sample = int(meta["n_sample"])
    seq_len = int(meta["seq_len"])
    top_k = int(sae_base.top_k)

    ids_path = os.path.join(llm_dir, "ids.dat")
    sae_dir = os.path.join(llm_dir, sae_base.sae_name, f"layer-{sae_base.layer}")
    idx_path = os.path.join(sae_dir, "sae_topk_idx.dat")
    vals_path = os.path.join(sae_dir, "sae_topk_val.dat")

    ids_mm = np.memmap(ids_path, dtype="int32", mode="r", shape=(n_sample, seq_len))
    idx_mm = np.memmap(idx_path, dtype="int32", mode="r", shape=(n_sample, seq_len, top_k))
    vals_mm = np.memmap(vals_path, dtype="float32", mode="r", shape=(n_sample, seq_len, top_k))

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    total_counts = _count_all_token_ids(ids_mm, vocab_size=vocab_size, chunk_samples=chunk_samples)

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
        os.path.join(sae_dir, "attribute", "output_sensitivity", "sensitivity.pt"), map_location="cpu"
    )

    # Register hooks
    embed_module = model_base._get_model_embed_modules()
    embed_module.weight.requires_grad_(True)

    residual_in: Dict[str, torch.Tensor] = {}

    def pre_hook_resid(module, inputs):
        residual_in["value"] = inputs[0]

    hook_handle = model_base.model_block_modules[int(sae_base.layer)].register_forward_pre_hook(
        pre_hook_resid
    )

    embed_out: Dict[str, torch.Tensor] = {}

    def hook_embed(module, inputs, output):
        embed_out["value"] = output
        output.retain_grad()

    embed_handle = embed_module.register_forward_hook(hook_embed)

    norm = model_base._get_model_norm_modules()
    lm_head = model_base._get_model_lm_head().weight

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    device = embed_module.weight.device

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

        act_counts = np.zeros((vocab_size,), dtype=np.int64)
        active_total = 0
        heap: List[Tuple[float, int]] = []

        for s0 in range(0, n_sample, chunk_samples):
            s1 = min(s0 + chunk_samples, n_sample)
            ids_chunk = np.asarray(ids_mm[s0:s1], dtype=np.int64)
            idx_chunk = np.asarray(idx_mm[s0:s1], dtype=np.int64)
            vals_chunk = np.asarray(vals_mm[s0:s1], dtype=np.float32)

            mask = idx_chunk == feature_idx
            any_mask = mask.any(axis=-1)

            if not np.any(any_mask):
                continue

            active_total += int(any_mask.sum())

            tok_ids = ids_chunk[any_mask].reshape(-1)
            valid = (tok_ids >= 0) & (tok_ids < vocab_size)
            if np.any(valid):
                act_counts += np.bincount(tok_ids[valid], minlength=vocab_size)

            feat_vals_pos = (vals_chunk * mask).sum(axis=-1)
            flat_vals = feat_vals_pos[any_mask].astype(np.float32, copy=False)
            flat_local = np.flatnonzero(any_mask)

            local_sample = (flat_local // seq_len).astype(np.int64)
            local_pos = (flat_local % seq_len).astype(np.int64)
            global_flat = (s0 + local_sample) * seq_len + local_pos

            for v, gi in zip(flat_vals.tolist(), global_flat.tolist()):
                if len(heap) < text_k:
                    heapq.heappush(heap, (float(v), int(gi)))
                elif float(v) > heap[0][0]:
                    heapq.heapreplace(heap, (float(v), int(gi)))

        with open(token_out_path, "a", encoding="utf-8") as f:
            f.write(f"activate / total: {active_total} / {total_positions}\n")
            entries = []
            for token_id in np.nonzero(act_counts)[0].tolist():
                if token_id in (bos_id, eos_id, pad_id):
                    continue
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

        for act_val, global_flat in heap_sorted:
            sample_idx = int(global_flat // seq_len)
            pos_idx = int(global_flat % seq_len)

            start = max(0, pos_idx - context_window_forward)
            end = min(seq_len, pos_idx + context_window_backward)
            window_ids = np.asarray(ids_mm[sample_idx, start:end], dtype=np.int64)

            prefix_ids_np = np.asarray(ids_mm[sample_idx, : pos_idx + 1], dtype=np.int64)
            prefix_ids = (
                torch.from_numpy(prefix_ids_np).to(device=device, dtype=torch.long).unsqueeze(0)
            )
            attention_mask = _make_left_padding_attention_mask(prefix_ids, pad_id=pad_id).to(device)

            residual_in.clear()
            embed_out.clear()
            model_base.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                _ = model_base.model(input_ids=prefix_ids, attention_mask=attention_mask)
                acts = residual_in["value"]
                feat_bt = sae_base.encode(acts)[0][:, :, feature_idx]
                feat_bt[0, pos_idx].backward(retain_graph=False)
                token_grads = embed_out["value"].grad[0]
                grad_norms = token_grads.to(dtype=torch.float32).norm(dim=-1).detach().cpu().numpy()

            window_grad_norms = grad_norms[start : pos_idx + 1]
            tokens = tokenizer.convert_ids_to_tokens(window_ids.tolist())

            if window_grad_norms.size > 0:
                topg_k = min(top_grad_k, window_grad_norms.size)
                topg_local_idx = window_grad_norms.argsort()[-topg_k:][::-1]
            else:
                topg_local_idx = np.array([], dtype=np.int64)

            topg_tokens = [
                tokens[int(j)] for j in topg_local_idx.tolist() if 0 <= int(j) < len(tokens)
            ]
            for j in topg_local_idx.tolist():
                if 0 <= int(j) < len(tokens):
                    tokens[int(j)] = f"[{tokens[int(j)]}]"
            relative_pos = pos_idx - start
            if 0 <= relative_pos < len(tokens):
                tokens[relative_pos] = f"[TOKEN: {tokens[relative_pos]}]"

            context_text = tokenizer.convert_tokens_to_string(tokens)
            entry_json = {
                "text": context_text,
                "activation": float(act_val),
                "top_grad_tokens": [
                    {"token": tok, "grad": float(window_grad_norms[int(j)])}
                    for tok, j in zip(topg_tokens, topg_local_idx.tolist())
                ],
            }

            top_lines = ["["]
            for item in entry_json["top_grad_tokens"]:
                line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                top_lines.append("  " + line + ",")
            if len(top_lines) > 1:
                top_lines[-1] = top_lines[-1].rstrip(",")
            top_lines.append("]")
            top_block = "\n".join(top_lines)

            json_pretty = json.dumps(entry_json, ensure_ascii=False, indent=2)
            json_final = re.sub(
                r'"top_grad_tokens": \[[\s\S]*?\]',
                f'"top_grad_tokens": {top_block}',
                json_pretty,
            )
            with open(topk_out_path, "a", encoding="utf-8") as f:
                f.write(json_final + "\n")

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

    hook_handle.remove()
    embed_handle.remove()

    return result
