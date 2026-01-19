import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------- CONFIG --------
INPUT_CSV = "hackernews.csv"
OUTPUT_CSV = "hackernews_with_embeddings.csv"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE = 128
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_texts(texts, tokenizer, model):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    # Remove output if it already exists
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    reader = pd.read_csv(INPUT_CSV, chunksize=BATCH_SIZE)

    for chunk in tqdm(reader, desc="Embedding titles"):
        titles = chunk["title"].fillna("").astype(str).tolist()

        embeddings = embed_texts(titles, tokenizer, model)
        chunk["embedding"] = [json.dumps(e) for e in embeddings]

        chunk.to_csv(
            OUTPUT_CSV,
            mode="a",
            index=False,
            header=not os.path.exists(OUTPUT_CSV)
        )

        # Explicitly free memory
        del embeddings
        torch.cuda.empty_cache()

    print(f"Done. Streaming embeddings saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
