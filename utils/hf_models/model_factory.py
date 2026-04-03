import torch

from typing import Union

from utils.hf_models.model_base import ModelBase

def construct_model_base(model_path: str, model_name: str, device: str = "cuda:0") -> ModelBase:

    if 'qwen3' in model_path.lower():
        from utils.hf_models.classes.qwen3_5 import Qwen3_5Model
        return Qwen3_5Model(model_path, model_name, device=device)
    if 'qwen' in model_path.lower():
        from utils.hf_models.classes.qwen import QwenModel
        return QwenModel(model_path, model_name, device=device)
    if 'llama-3' in model_path.lower():
        from utils.hf_models.classes.llama3 import Llama3Model
        return Llama3Model(model_path, model_name, device=device)
    elif 'gemma' in model_path.lower():
        from utils.hf_models.classes.gemma import GemmaModel
        return GemmaModel(model_path, model_name, device=device)
    else:
        raise ValueError(f"Unknown model family: {model_path}")