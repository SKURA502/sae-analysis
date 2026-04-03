import torch
import functools

from jaxtyping import Float
from torch import Tensor
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.hf_models.model_base import ModelBase


QWEN3_5_TOOL_SYSTEM_TEMPLATE = """\
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

QWEN3_5_REFUSAL_TOKS = [40]  # 'I'


class Qwen3_5Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16, device: str = "cuda:0"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return None

    def _get_eoi_toks(self):
        return []

    def _get_refusal_toks(self):
        return QWEN3_5_REFUSAL_TOKS

    def _get_model_embed_modules(self):
        return self.model.model.embed_tokens

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        # Qwen3.5 is a hybrid model; some layers use linear_attn, others may differ.
        # This attribute is not used in sae-analysis but must be implemented.
        modules = []
        for block in self.model.model.layers:
            attn = getattr(block, 'linear_attn', None) or getattr(block, 'self_attn', None)
            if attn is not None:
                modules.append(attn)
        return torch.nn.ModuleList(modules)

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model.model.layers])

    def _get_model_norm_modules(self):
        return self.model.model.norm

    def _get_model_lm_head(self):
        return self.model.lm_head
