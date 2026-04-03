export PYTHONPATH=/data/sae-analysis:$PYTHONPATH

python -u main.py \
    --model_alias meta-llama/Llama-3.1-8B-Instruct \
    --model_path /data/source/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --sae_name Llama-Mixed \
    --sae_path /data/jailbreak-interpretability/saes/Llama3.1-8B-Mixed/layer-31/ae.pt \
    --top_k 64 \
    --layer 31 \
    --dtype bfloat16 \
    --norm_factor 72.38159367228012 \
    --device cuda:0

python -u main.py \
    --model_alias meta-llama/Llama-3.1-8B \
    --model_path /data/source/model/LLM-Research/Meta-Llama-3.1-8B \
    --sae_name Llama-Mixed \
    --sae_path /data/jailbreak-interpretability/saes/Llama3.1-8B-Mixed/ae.pt \
    --top_k 64 \
    --layer 16 \
    --dtype bfloat16 \
    --norm_factor 12.130116208770684 \
    --device cuda:1

python -u main.py \
    --model_alias meta-llama/Llama-3.1-8B \
    --model_path /data/source/model/LLM-Research/Meta-Llama-3.1-8B \
    --sae_name Llama-Scope \
    --sae_path /data/source/sae/Llama3.1-8B/Llama-Scope/Llama3_1-8B-Base-L30R-8x/checkpoints/final.safetensors \
    --top_k 50 \
    --layer 30 \
    --dtype bfloat16 \
    --norm_factor 53.25 \
    --device cuda:2

python -u main.py \
    --model_alias qwen3.5-4b \
    --model_path /data/source/model/Qwen/Qwen3.5-4B \
    --sae_name Tool-use \
    --sae_path /data/Agent-Tool-Use-MI/checkpoint/stage2/Qwen3.5-4B-L25-d20480-5M-stage2.pt\
    --top_k 80 \
    --layer 25 \
    --dtype bfloat16 \
    --device cuda:2 \
    --second_dataset when2call