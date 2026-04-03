import json
import random
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice

BEAVERTAILS_JSONL  = "/data/source/dataset/BeaverTails/round0/330k/responses_llm.jsonl"
GEN_RESPONSE_KEY   = "Llama3.1-8B-Instruct-Response"

WHEN2CALL_JSONL    = "/data/source/dataset/when2call/train/when2call_train_pref.jsonl"

from utils.hf_models.classes.qwen3_5 import QWEN3_5_TOOL_SYSTEM_TEMPLATE

def load_fineweb_text_to_cache(model_base, n_sample):
    print("[FineWeb] Loading LOCAL FineWeb dataset...")

    dataset = load_dataset(
        "/data/source/dataset/HuggingFaceFW___fineweb/sample-10BT",
        split="train",
        streaming=True,
    )

    texts = []
    for ex in tqdm(islice(dataset, int(n_sample)), total=int(n_sample), desc="FineWeb"):
        texts.append(ex["text"])

    return texts


def load_beavertails_text_to_cache(model_base, n_sample):
    print(f"[BeaverTails] Loading from {BEAVERTAILS_JSONL} ...")

    tokenizer = model_base.tokenizer
    texts = []
    with open(BEAVERTAILS_JSONL) as f:
        for line in tqdm(f, total=int(n_sample), desc="BeaverTails"):
            if len(texts) >= int(n_sample):
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt, response = d["prompt"], d[GEN_RESPONSE_KEY]
            try:
                conv = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}]
                text = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"User: {prompt}\n\nAssistant: {response}"
            texts.append(text)

    return texts


def load_when2call_text_to_cache(model_base, n_sample):
    print(f"[When2Call] Loading from {WHEN2CALL_JSONL} ...")

    with open(WHEN2CALL_JSONL) as f:
        total_lines = sum(1 for _ in f)
    actual_total = min(int(n_sample), total_lines)
    print(f"[When2Call] {total_lines} lines in file, requesting {int(n_sample)}, will load {actual_total}")

    tokenizer = model_base.tokenizer
    texts = []
    with open(WHEN2CALL_JSONL) as f:
        for line in tqdm(f, total=actual_total, desc="When2Call"):
            if len(texts) >= int(n_sample):
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            tools: list = d.get("tools") or []
            tool_strs = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in tools]
            tools_str = "\n".join(tool_strs) if tool_strs else "(none)"

            messages_raw: list = d.get("messages") or []
            user_content = next(
                (m["content"] for m in messages_raw if m.get("role") == "user"),
                "(no question)",
            )

            chosen = d.get("chosen_response") or {}
            assistant_content = chosen.get("content", "") if isinstance(chosen, dict) else str(chosen)

            conv = [
                {"role": "system",    "content": QWEN3_5_TOOL_SYSTEM_TEMPLATE.format(tools_str=tools_str)},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = "\n\n".join(m["content"] for m in conv)
            texts.append(text)

    return texts


def load_mixed_text_to_cache(model_base, n_sample, second_dataset="beavertails"):
    half = int(n_sample) // 2
    fineweb_texts = load_fineweb_text_to_cache(model_base, half)
    if second_dataset == "when2call":
        second_texts = load_when2call_text_to_cache(model_base, half)
    else:
        second_texts = load_beavertails_text_to_cache(model_base, half)
    texts = fineweb_texts + second_texts
    random.shuffle(texts)
    return texts