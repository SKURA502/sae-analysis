import os
import torch

from safetensors.torch import load_file

class SAEBase():
    def __init__(self, sae_path, **kwargs):
        self.normalize_acts = kwargs.get("normalize_acts", False)
        self.device = kwargs.get("device", "cpu")
        self.top_k = kwargs.get("top_k", None)
        self.sae_name = kwargs.get("name", None)
        self.layer = kwargs.get("layer", 0)
        self.dtype = getattr(torch, kwargs.get("dtype", "float32"))

        self.sae = self.load_sae(sae_path)
        self.act_dim = self.sae['W_enc'].shape[0]
        self.sae_dim = self.sae['W_enc'].shape[1]

        if self.top_k is None:
            self.top_k = self.sae_dim // 100
        
        token_freq_path = kwargs.get("token_freq", None)
        self.token_freq = torch.load(token_freq_path, device=self.device) if token_freq_path is not None else None

    def load_sae(self, sae_path):
        """
        Load SAE weights from .pt or .safetensors and
        normalize key names to:
            W_enc, b_enc, W_dec, b_dec
        """
        # ---------- 1. load raw state dict ----------
        ext = os.path.splitext(sae_path)[1]

        if ext in [".pt", ".pth"]:
            state = torch.load(sae_path, map_location="cpu")
        elif ext == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(sae_path)
        else:
            raise ValueError(f"Unsupported SAE file format: {ext}")
        
        key_map = {
            "W_enc": ["W_enc", "encoder.weight", "enc.weight"],
            "b_enc": ["b_enc", "encoder.bias", "enc.bias"],
            "W_dec": ["W_dec", "decoder.weight", "dec.weight"],
            "b_dec": ["b_dec", "decoder.bias", "dec.bias", "bias"],
        }

        sae = {}

        for canonical_key, aliases in key_map.items():
            found = False
            for k in aliases:
                if k in state:
                    tensor = state[k]

                    if canonical_key == "W_enc":
                        if tensor.ndim >= 2 and tensor.shape[0] > tensor.shape[1]:
                            tensor = tensor.transpose(0, 1)
                    elif canonical_key == "W_dec":
                        if tensor.ndim >= 2 and tensor.shape[0] < tensor.shape[1]:
                            tensor = tensor.transpose(0, 1)

                    sae[canonical_key] = tensor.to(
                        dtype=self.dtype,
                        device=self.device
                    )
                    found = True
                    break

            if not found:
                raise KeyError(
                    f"Could not find {canonical_key}. "
                    f"Tried aliases: {aliases}. "
                    f"Available keys: {list(state.keys())}"
                )
        return sae

    def encode(self, activation):
        activation = activation.to(dtype=self.dtype)
        if self.normalize_acts:
            input_norm = torch.norm(activation, p=2, dim=-1, keepdim=True)
            activation = activation / input_norm * torch.sqrt(torch.tensor(activation.size(-1), dtype=torch.float32, device=self.device))
            
        W_enc = self.sae['W_enc']
        b_enc = self.sae['b_enc']

        latent = torch.einsum("bld,dh->blh", activation, W_enc) + b_enc
        latent_relu = torch.relu(latent)

        topk_vals, topk_idx = torch.topk(latent_relu, self.top_k, dim=-1)

        return latent_relu, topk_vals.to(dtype=torch.float), topk_idx


