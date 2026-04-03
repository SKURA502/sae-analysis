import os
import torch

from safetensors.torch import load_file

class SAEBase():
    def __init__(self, sae_path, **kwargs):
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

        _norm_factor = kwargs.get("norm_factor", None)
        self.norm_factor = float(_norm_factor) if _norm_factor is not None else self.act_dim ** 0.5

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
            import sys, types
            # The .pt file may have been saved with a custom `sae` package that is
            # not available here.  Inject a lightweight stub so pickle can resolve
            # any embedded classes (e.g. sae.sae_model.SAEConfig) without error.
            if "sae" not in sys.modules:
                _stub_sae = types.ModuleType("sae")
                _stub_sae_model = types.ModuleType("sae.sae_model")
                class _AnyConfig:
                    def __init__(self, *a, **kw): pass
                    def __setstate__(self, state): self.__dict__.update(state)
                _stub_sae_model.SAEConfig = _AnyConfig
                _stub_sae.sae_model = _stub_sae_model
                sys.modules["sae"] = _stub_sae
                sys.modules["sae.sae_model"] = _stub_sae_model
            state = torch.load(sae_path, map_location="cpu", weights_only=False)
            # If the file was saved as a full model object rather than a state dict,
            # extract the underlying state dict.
            if not isinstance(state, dict):
                state = state.state_dict() if hasattr(state, "state_dict") else vars(state)
            # Handle checkpoint format: {'config': ..., 'state_dict': {...}}
            if "state_dict" in state and not any(k in state for k in ("W_enc", "encoder.weight", "enc.weight")):
                state = state["state_dict"]
        elif ext == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(sae_path)
        else:
            raise ValueError(f"Unsupported SAE file format: {ext}")
        
        # Required keys
        required_map = {
            "W_enc": ["W_enc", "encoder.weight", "enc.weight"],
            "W_dec": ["W_dec", "decoder.weight", "dec.weight"],
        }
        # Optional keys: missing ones are stored as None
        optional_map = {
            "b_enc":     ["b_enc", "encoder.bias", "enc.bias"],
            "b_dec":     ["b_dec", "decoder.bias", "dec.bias", "bias"],
            "pre_bias":  ["pre_bias"],
        }

        sae = {}

        for canonical_key, aliases in required_map.items():
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
                    sae[canonical_key] = tensor.to(dtype=self.dtype, device=self.device)
                    found = True
                    break
            if not found:
                raise KeyError(
                    f"Could not find {canonical_key}. "
                    f"Tried aliases: {aliases}. "
                    f"Available keys: {list(state.keys())}"
                )

        for canonical_key, aliases in optional_map.items():
            sae[canonical_key] = None
            for k in aliases:
                if k in state:
                    sae[canonical_key] = state[k].to(dtype=self.dtype, device=self.device)
                    break

        # Fold pre_bias into b_enc / b_dec so all models use a unified bias format.
        # pre_bias is subtracted from input before W_enc, equivalent to:
        #   b_enc -= pre_bias @ W_enc
        #   b_dec  = pre_bias
        if sae['pre_bias'] is not None:
            contrib = torch.einsum("d,dh->h", sae['pre_bias'], sae['W_enc'])
            sae['b_enc'] = (sae['b_enc'] if sae['b_enc'] is not None else torch.zeros_like(contrib)) - contrib
            sae['b_dec'] = (sae['b_dec'] if sae['b_dec'] is not None else torch.zeros_like(sae['pre_bias'])) + sae['pre_bias']
            sae['pre_bias'] = None

        return sae

    def encode(self, activation):
        activation = activation.to(dtype=self.dtype)
        activation = activation / self.norm_factor * (self.act_dim ** 0.5)

        W_enc = self.sae['W_enc']

        latent = torch.einsum("bld,dh->blh", activation, W_enc)

        if self.sae['b_enc'] is not None:
            latent = latent + self.sae['b_enc']

        latent_relu = torch.relu(latent)
        topk_vals, topk_idx = torch.topk(latent_relu, self.top_k, dim=-1)

        return latent_relu, topk_vals.to(dtype=torch.float), topk_idx


