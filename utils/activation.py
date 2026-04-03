import os
import torch
import json
import numpy as np

from torch import Tensor
from typing import List, Tuple, Union
from jaxtyping import Int, Float
from tqdm import tqdm, trange

from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.sae.sae_base import SAEBase

CHUNK_BATCHES = 156

def _get_activations_pre_hook(cache: Float[Tensor, "pos d_model"]):
    def hook_fn(module, input):
        nonlocal cache
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, :] += activation[:, :].to(cache)
    return hook_fn

def _get_activations_fwd_hook(cache: Float[Tensor, "pos d_model"]):
    def hook_fn(module, input, output):
        nonlocal cache
        activation: Float[Tensor, "batch_size, seq_len, d_model"] = output[0].clone().to(cache) if isinstance(output, tuple) else output.clone().to(cache)
        cache[:, :] += activation[:, :].to(cache)
    return hook_fn

@torch.no_grad()
def _get_activations_fixed_seq_len(
    model, tokenizer, prompts: List[str], block_modules: List[torch.nn.Module], 
    seq_len: int = 512, layers: List[int]=None, batch_size=32, 
    save_device: Union[torch.device, str] = "cuda", verbose=True
) -> Tuple[Float[Tensor, 'n seq_len'], Float[Tensor, 'n n_layers seq_len d_model']]:
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    n_layers = len(layers)
    d_model = model.config.hidden_size

    # we store the activations in high-precision to avoid numerical issues
    activations = torch.zeros((len(prompts), n_layers, seq_len, d_model), device=save_device)
    all_input_ids = torch.zeros((len(prompts), seq_len), dtype=torch.long, device=save_device)

    # Fill all_input_ids with tokenizer.pad_token_id
    all_input_ids.fill_(tokenizer.pad_token_id)

    for i in tqdm(range(0, len(prompts), batch_size), disable=not verbose):
        inputs = tokenizer(prompts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=seq_len)

        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        inputs_len = len(input_ids)
        num_input_toks = input_ids.shape[-1]

        fwd_hooks = [
            (block_modules[layer], _get_activations_fwd_hook(cache=activations[i:i+inputs_len, layer_idx, -num_input_toks:, :])) 
            for layer_idx, layer in enumerate(layers)
        ]

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        all_input_ids[i:i+inputs_len, -num_input_toks:] = input_ids

    return all_input_ids, activations

def compute_activation(model_base: ModelBase, prompts: List[str], batch_size: int, seq_len: int):

    n_layers = model_base.model.config.num_hidden_layers
    layers = range(n_layers)

    input_ids, activations  = _get_activations_fixed_seq_len(
        model_base.model,
        model_base.tokenizer,
        prompts=prompts,
        block_modules=model_base.model_block_modules,
        seq_len=seq_len,
        layers=layers,
        batch_size=batch_size,
        save_device=model_base.model.device,
        verbose=False
    )
    activations: Float[Tensor, 'n n_layers seq_len d_model'] = activations.cpu()
    input_ids: Int[Tensor, 'n seq_len'] = input_ids.cpu()

    return input_ids, activations
                
def get_activation(model_base: ModelBase, sae_base: SAEBase, prompts: List[str], 
                    batch_size: int = 64, seq_len: int = 128):

    d_model  = model_base.model.config.hidden_size
    n_sample = len(prompts)
    n_layers = model_base.model.config.num_hidden_layers

    print('shape act size', (n_sample, n_layers, seq_len, d_model))

    foldername = f"./data/{model_base.model_name}/{sae_base.sae_name}"
    sae_foldername = os.path.join(foldername, f"layer-{sae_base.layer}")
    if os.path.exists(sae_foldername):
        print(f"[Cache hit] Found existing sae activations at {sae_foldername}")
        return
    os.makedirs(sae_foldername, exist_ok=True)

    memmap_file_ids = np.memmap(
        os.path.join(foldername, f"ids.dat"),
        dtype='int32',
        mode='w+',
        shape=(n_sample, seq_len)
    )

    top_k = sae_base.top_k
    vals_mm = np.memmap(
        os.path.join(sae_foldername, "sae_topk_val.dat"),
        dtype="float32",
        mode="w+",
        shape=(n_sample, seq_len, top_k)
    )
    idx_mm = np.memmap(
        os.path.join(sae_foldername, "sae_topk_idx.dat"),
        dtype="int32",
        mode="w+",
        shape=(n_sample, seq_len, top_k)
    )

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing activations and SAE"):
        end_id = min(i + batch_size, len(prompts))
        batch_prompts = prompts[i:end_id]

        input_ids, activations = compute_activation(
            model_base=model_base,
            prompts=batch_prompts,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        memmap_file_ids[i:end_id] = input_ids.cpu().numpy()

        # Directly process SAE for this batch
        acts = activations[:, sae_base.layer, :, :]  # (batch_size, seq_len, d_model)
        _, topk_vals, topk_idx = sae_base.encode(acts.to(sae_base.device))

        vals_mm[i:end_id] = topk_vals.cpu().numpy()
        idx_mm[i:end_id] = topk_idx.cpu().numpy()

    vals_mm.flush()
    idx_mm.flush()

    meta = {
        "dtype": "float32",
        "model": model_base.model_name,
        "n_sample": n_sample,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "d_model": d_model,
        "batch_size": batch_size,
        "top_k": top_k,
        "sae_layer": sae_base.layer,
        "sae_name": sae_base.sae_name,
    }

    with open(os.path.join(foldername, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
