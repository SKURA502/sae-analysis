import os
import json
from typing import Dict

import numpy as np

from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase

def _load_token_flag_lookup(tokenizer, token_flag_map_path: str) -> np.ndarray:
    """
    Build a dense lookup table flag[token_id] in {0,1}.
    Unknown tokens default to 0.
    """
    with open(token_flag_map_path, "r", encoding="utf-8") as f:
        token_flag_map = json.load(f)

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    lookup = np.zeros((vocab_size,), dtype=np.uint8)
    for k, v in token_flag_map.items():
        try:
            tid = int(k)
        except Exception:
            continue
        if 0 <= tid < vocab_size:
            lookup[tid] = 1 if int(v) == 1 else 0

    return lookup


def _generate_token_flag_map_from_ids(
    ids_mm: np.memmap,
    tokenizer,
    coverage: float = 0.40,
    chunk_samples: int = 2048,
) -> dict:
    """Generate token_flag_map according to token frequency.

    Rule:
      - Compute token empirical probability from ids.dat.
      - Sort tokens by probability (descending).
      - Set flag=1 for tokens whose cumulative probability reaches the first
        `coverage` fraction (default 0.40). Others are 0.

    Notes:
      - Excludes special tokens (bos/eos/pad) from counting.
      - Unknown/negative ids are ignored.
    """
    if not (0.0 < float(coverage) < 1.0):
        raise ValueError(f"coverage must be in (0,1), got {coverage}")

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    counts = np.zeros((vocab_size,), dtype=np.int64)
    total_valid = 0

    n_sample, seq_len = ids_mm.shape
    for start in range(0, n_sample, chunk_samples):
        end = min(start + chunk_samples, n_sample)
        ids = np.asarray(ids_mm[start:end])  # (B, L)
        flat = ids.reshape(-1)

        valid = (flat >= 0) & (flat < vocab_size)
        if bos_id is not None:
            valid &= (flat != int(bos_id))
        if eos_id is not None:
            valid &= (flat != int(eos_id))
        if pad_id is not None:
            valid &= (flat != int(pad_id))

        if not np.any(valid):
            continue

        flat_valid = flat[valid].astype(np.int64, copy=False)
        total_valid += int(flat_valid.size)
        counts += np.bincount(flat_valid, minlength=vocab_size)

    if total_valid <= 0:
        raise RuntimeError("No valid tokens found in ids.dat to build token_flag_map.")

    probs = counts / float(total_valid)
    order = np.argsort(-probs)  # descending
    probs_sorted = probs[order]
    cum = np.cumsum(probs_sorted)

    # Include tokens up to the first index where cumulative >= coverage
    cutoff_idx = int(np.searchsorted(cum, float(coverage), side="left"))
    cutoff_idx = min(cutoff_idx, vocab_size - 1)
    selected = order[: cutoff_idx + 1]

    flags = np.zeros((vocab_size,), dtype=np.uint8)
    flags[selected] = 1

    # Export as {"token_id": 0/1} dict (string keys)
    return {str(i): int(flags[i]) for i in range(vocab_size)}


def _ensure_token_flag_map(
    model_base: ModelBase,
    ids_mm: np.memmap,
    save_dir: str,
    coverage: float = 0.40,
) -> str:
    """Ensure token_flag_map.json exists in save_dir.

    Priority:
      1) If {save_dir}/token_flag_map.json exists -> use it.
      2) Else -> generate from ids.dat (top cumulative `coverage`) and save into save_dir.
    """
    local_path = os.path.join(save_dir, "token_flag_map.json")
    if os.path.exists(local_path):
        return local_path

    os.makedirs(save_dir, exist_ok=True)
    token_flag_map = _generate_token_flag_map_from_ids(
        ids_mm=ids_mm,
        tokenizer=model_base.tokenizer,
        coverage=float(coverage),
    )
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(token_flag_map, f)
    return local_path