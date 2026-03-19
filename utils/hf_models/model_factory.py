import torch

from typing import Union

from utils.hf_models.model_base import ModelBase

def construct_model_base(model_path: str, model_name: str) -> ModelBase:

    if 'qwen' in model_path.lower():
        from utils.hf_models.classes.qwen import QwenModel
        return QwenModel(model_path, model_name)
    if 'llama-3' in model_path.lower():
        from utils.hf_models.classes.llama3 import Llama3Model
        return Llama3Model(model_path, model_name)
    elif 'gemma' in model_path.lower():
        from utils.hf_models.classes.gemma import GemmaModel
        return GemmaModel(model_path, model_name) 
    else:
        raise ValueError(f"Unknown model family: {model_path}")