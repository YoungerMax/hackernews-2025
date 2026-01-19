import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------- CONFIG --------
INPUT_CSV = "hackernews_with_embeddings.csv"
OUTPUT_CSV = "hackernews_with_tags.csv"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

TAGS = [
    "startups",
    "big tech",
    "open source",
    "programming languages",
    "developer tools",
    "artificial intelligence",
    "data engineering",
    "cybersecurity",
    "operating systems",
    "science research",
    "math and theory",
    "economics and markets",
    "venture capital",
    "geopolitics",
    "public policy",
    "privacy and surveillance",
    "digital culture",
    "productivity and careers",
    "education and learning",
    "job postings"
]

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
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)

    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    print("Embedding tags...")
    tag_embeddings = embed_texts(TAGS, tokenizer, model)  # shape: (20, dim)

    print("Classifying posts...")
    assigned_tags = []

    for emb_json in tqdm(df["embedding"], desc="Assigning tags"):
        post_emb = np.array(json.loads(emb_json), dtype=np.float32)

        # cosine similarity since all embeddings are normalized
        sims = tag_embeddings @ post_emb
        best_tag_idx = int(np.argmax(sims))
        assigned_tags.append(TAGS[best_tag_idx])

    df["tag"] = assigned_tags

    print("Ranking tags by frequency...")

    tag_counts = (
        df["tag"]
        .value_counts()
        .reset_index()
    )

    tag_counts.columns = ["tag", "count"]

    print("\n=== TAG RANKING (DESCENDING) ===")
    for _, row in tag_counts.iterrows():
        print(f"{row['tag']:25s} {row['count']}")


    print("\nSaving output CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
