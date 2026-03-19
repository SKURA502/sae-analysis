import sys
sys.path.append('../')
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse

import attribute.functional_ratio.main as functional_ratio
import attribute.entropy.main as entropy
import attribute.activation_length.main as activation_length
import attribute.activation_value.main as activation_value
import attribute.output_sensitivity.main as output_sensitivity

from dataset.load import load_fineweb_text_to_cache

from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from utils.activation import get_activation
from utils.sae.sae_base import SAEBase

def Args():
    parser = argparse.ArgumentParser(description='Cache activations for a given model')
    parser.add_argument("--model_alias", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--sae_name", type=str, default="Llama-Scope")
    parser.add_argument("--sae_path", type=str, default=None)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--normalize_acts", action="store_true")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--layer", type=int, default=30)

    return parser.parse_args()

if __name__ == "__main__":
    args = Args()

    # Load Models
    model_alias = args.model_alias
    model_name = model_alias_to_model_name[model_alias]

    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = model_alias_to_model_name[model_alias]
    model_base = construct_model_base(model_path, model_name)

    # Load SAE
    sae_cfg = {
        "device": args.device,
        "normalize_acts": args.normalize_acts,
        "top_k": args.top_k,
        "layer": args.layer,
        "name": args.sae_name,
        "dtype": args.dtype
    }
    sae_base = SAEBase(args.sae_path, **sae_cfg)

    # Get text activation on all layers
    texts = load_fineweb_text_to_cache(model_base, n_sample=1e5)
    get_activation(model_base, sae_base, texts, batch_size=64, seq_len=128)

    selected_eval = [
        # "functional_ratio",
        # "entropy",
        # "activation_length",
        # "activation_value",
        "output_sensitivity",
    ]

    if "functional_ratio" in selected_eval:
        functional_ratio.run(
            model_base=model_base,
            sae_base=sae_base,
        )
    
    if "entropy" in selected_eval:
        entropy.run(
            model_base=model_base, 
            sae_base=sae_base
        )

    if "activation_length" in selected_eval:
        activation_length.run(
            model_base=model_base, 
            sae_base=sae_base
        )

    if "activation_value" in selected_eval:
        activation_value.run(
            model_base=model_base, 
            sae_base=sae_base
        )

    if "output_sensitivity" in selected_eval:
        output_sensitivity.run(
            model_base=model_base,
            sae_base=sae_base,
        )


    





    

