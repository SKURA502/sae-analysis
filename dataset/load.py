from datasets import load_dataset
from tqdm import tqdm
from itertools import islice

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