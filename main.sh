export PYTHONPATH=/data/sae-analysis:$PYTHONPATH

nohup python -u main.py \
    --model_alias meta-llama/Llama-3.1-8B \
    --model_path /data/source/model/LLM-Research/Meta-Llama-3.1-8B \
    --sae_name Llama-Scope \
    --sae_path /data/source/sae/Llama3.1-8B/Llama-Scope/Llama3_1-8B-Base-L30R-8x/checkpoints/final.safetensors \
    --normalize_acts \
    --top_k 50 \
    --layer 30 \
    --dtype bfloat16 \
    > log/main.log 2>&1 &

# nohup python main.py \
#     --model_alias meta-llama/Llama-3.1-8B \
#     --model_path /mindopt/project/source/model/Llama/Llama-3.1-8B \
#     --sae_name custom-standard \
#     --sae_path /mindopt/project/source/sae/Llama/Llama-3.1-8B/custom-standard/resid_post_layer_30/trainer_0/ae.pt \
#     --top_k 100 \
#     --layer 30 \
#     --dtype float \
#     > log/custom_standard_main.log 2>&1 &