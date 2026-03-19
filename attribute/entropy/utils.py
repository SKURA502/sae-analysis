from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def decoder_entropy(
    decoder_vec: torch.Tensor,          # [B, d_model] or [d_model]
    norm: torch.nn.Module,
    lm_head: torch.Tensor,        # [vocab, d_model]
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if decoder_vec.dim() == 1:
        decoder_vec = decoder_vec.unsqueeze(0)  # -> [1, d_model]
    assert decoder_vec.dim() == 2, f"decoder_vec must be [B,d], got {decoder_vec.shape}"

    x = norm(decoder_vec).to(torch.float32)                 # [B, d_model]
    W = lm_head.to(torch.float32)                     # [vocab, d_model]
    logits = x @ W.T                                         # [B, vocab]

    probs = F.softmax(logits, dim=-1)[0]                # [B, vocab]
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

    topk_ids = topk_probs = None
    if topk is not None:
        topk_probs, topk_ids = torch.topk(probs, k=int(topk), dim=-1)  # [B, topk], long

    return entropy, topk_ids, topk_probs