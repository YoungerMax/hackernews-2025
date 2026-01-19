#!/usr/bin/env python3
"""
Reworked clustering pipeline:
- Load embeddings from CSV (streaming)
- Standard scale
- PCA -> UMAP (for clustering) -> HDBSCAN (on UMAP low-dim)
- UMAP (2D) for visualization / spatial chunking
- Summarize clusters using Ollama (sample up to SAMPLE_K titles per cluster)
- Save spatial chunks + metadata
"""

import os
import json
import random
import multiprocessing

import orjson
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from ollama import Client

# -----------------------------
# CONFIG (tweak these)
# -----------------------------
CSV_PATH = "hackernews_with_embeddings.csv"
EMBEDDING_COL = "embedding"
ID_COL = "id"
TITLE_COL = "title"

# Dimensionality reduction
PCA_COMPONENTS = 30                  # reduce to ~30 before UMAP
UMAP_CLUSTER_COMPONENTS = 10         # UMAP dims used for clustering (lower = easier to cluster)
UMAP_CLUSTER_N_NEIGHBORS = 30
UMAP_CLUSTER_MIN_DIST = 0.0
UMAP_CLUSTER_METRIC = "cosine"

# UMAP for visualization (2D)
UMAP_2D_N_NEIGHBORS = 15
UMAP_2D_MIN_DIST = 0.1
UMAP_2D_METRIC = "cosine"

# HDBSCAN clustering
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 5
CLUSTER_METRIC = "euclidean"            # try 'cosine' for embeddings
CORE_DIST_N_JOBS = multiprocessing.cpu_count()

# Ollama / cluster summarization
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:27b-cloud"
SAMPLE_K = 50                       # max number of titles to sample for summary

# Output
OUTPUT_DIR = "hn_clusters_viewer"
SPATIAL_GRID_SIZE = 10               # NxN grid for chunking

# Misc
RANDOM_STATE = 42
CHUNKSIZE = 50_000

# -----------------------------
# LOAD DATA (streaming)
# -----------------------------
print("Loading CSV (streaming)...")

embeddings = []
titles = []
ids = []

for chunk in pd.read_csv(CSV_PATH, usecols=[ID_COL, TITLE_COL, EMBEDDING_COL], chunksize=CHUNKSIZE):
    for i, row in chunk.iterrows():
        try:
            if pd.isna(row[TITLE_COL]) or not isinstance(row[TITLE_COL], str):
                continue
            emb = orjson.loads(row[EMBEDDING_COL])
            embeddings.append(emb)
            titles.append(row[TITLE_COL])
            ids.append(int(row[ID_COL]))
        except Exception:
            # skip malformed rows
            continue

if len(embeddings) == 0:
    raise SystemExit("No embeddings loaded. Check CSV path and columns.")

print(f"Loaded {len(embeddings)} embeddings")

X = np.array(embeddings, dtype=np.float32)
del embeddings  # free memory

# -----------------------------
# NORMALIZE
# -----------------------------
print("Scaling embeddings...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA
# -----------------------------
print(f"Running PCA -> {PCA_COMPONENTS} components...")
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# UMAP (for clustering) - low dim (e.g. 10D)
# -----------------------------
print(f"Running UMAP for clustering -> {UMAP_CLUSTER_COMPONENTS} components (n_neighbors={UMAP_CLUSTER_N_NEIGHBORS}, min_dist={UMAP_CLUSTER_MIN_DIST})...")
umap_cluster = umap.UMAP(
    n_neighbors=UMAP_CLUSTER_N_NEIGHBORS,
    min_dist=UMAP_CLUSTER_MIN_DIST,
    n_components=UMAP_CLUSTER_COMPONENTS,
    metric=UMAP_CLUSTER_METRIC,
    random_state=RANDOM_STATE,
)
X_umap_cluster = umap_cluster.fit_transform(X_pca)

# -----------------------------
# HDBSCAN CLUSTERING (on UMAP low-dim)
# -----------------------------
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric=CLUSTER_METRIC,
    cluster_selection_method="eom",
    core_dist_n_jobs=CORE_DIST_N_JOBS
)

labels = clusterer.fit_predict(X_umap_cluster)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")

# -----------------------------
# UMAP (2D visualization)
# -----------------------------
print("Running UMAP (2D) for visualization and spatial chunking...")
umap_2d = umap.UMAP(
    n_neighbors=UMAP_2D_N_NEIGHBORS,
    min_dist=UMAP_2D_MIN_DIST,
    n_components=2,
    metric=UMAP_2D_METRIC,
    random_state=RANDOM_STATE
)
X_umap_2d = umap_2d.fit_transform(X_pca)  # use same PCA input for visual consistency

# -----------------------------
# Summarize clusters using Ollama
# -----------------------------
print("Summarizing clusters using Ollama...")
client = Client(host=OLLAMA_HOST)

cluster_info = {}
unique_labels = sorted(set(labels))

for cluster_id in unique_labels:
    mask = labels == cluster_id
    count = int(np.sum(mask))
    if cluster_id == -1:
        cluster_info[-1] = {"id": -1, "name": "Noise", "count": count}
        print(f"  Cluster -1 (Noise): {count} posts")
        continue

    # collect titles
    cluster_titles = [titles[i] for i in range(len(titles)) if mask[i]]

    # sample titles (without replacement if possible)
    sample_k = min(SAMPLE_K, len(cluster_titles))
    if len(cluster_titles) <= sample_k:
        sample_titles = cluster_titles
    else:
        sample_titles = random.sample(cluster_titles, sample_k)

    prompt = (
        "Given these Hacker News post titles, provide ONLY a short 2-5 word cluster name. "
        "Do not include quotes, asterisks, or any explanation. Just the name.\n\n"
        f"Titles:\n{sample_titles}\n\n"
        "Cluster name:"
    )

    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cluster_name = response.message.content.strip()
    except Exception as e:
        # fallback: simple automatic label if Ollama fails
        print(f"    Ollama error for cluster {cluster_id}: {e!r} — using fallback name.")
        # naive fallback: first 3 words of the most common title
        most_common_title = max(cluster_titles, key=lambda t: len(t))
        cluster_name = " ".join(most_common_title.split()[:4])

    # Cleanup
    cluster_name = cluster_name.replace('"', '').replace('*', '').replace('\n', ' ')
    cluster_name = ' '.join(cluster_name.split())

    cluster_info[int(cluster_id)] = {"id": int(cluster_id), "name": cluster_name, "count": count}
    print(f"  Cluster {cluster_id}: {cluster_name} ({count} posts)")

# -----------------------------
# SAVE DATA AS SPATIAL CHUNKS
# -----------------------------
print(f"\nSaving data to {OUTPUT_DIR}/ ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filter out noise for spatial chunking (if you want to include noise remove this line)
non_noise_mask = labels != -1
filtered_indices = np.where(non_noise_mask)[0]

# Coordinates from 2D UMAP for visualization/chunking
x_coords = X_umap_2d[filtered_indices, 0]
y_coords = X_umap_2d[filtered_indices, 1]
x_min, x_max = float(x_coords.min()), float(x_coords.max())
y_min, y_max = float(y_coords.min()), float(y_coords.max())

# Add small padding
x_range = x_max - x_min
y_range = y_max - y_min
if x_range == 0: x_range = 1.0
if y_range == 0: y_range = 1.0
x_min -= x_range * 0.05
x_max += x_range * 0.05
y_min -= y_range * 0.05
y_max += y_range * 0.05

grid_width = (x_max - x_min) / SPATIAL_GRID_SIZE
grid_height = (y_max - y_min) / SPATIAL_GRID_SIZE

spatial_chunks = {}

for idx in filtered_indices:
    x = float(X_umap_2d[idx, 0])
    y = float(X_umap_2d[idx, 1])

    grid_x = int((x - x_min) / grid_width)
    grid_y = int((y - y_min) / grid_height)
    grid_x = max(0, min(SPATIAL_GRID_SIZE - 1, grid_x))
    grid_y = max(0, min(SPATIAL_GRID_SIZE - 1, grid_y))
    chunk_key = f"{grid_x}_{grid_y}"
    spatial_chunks.setdefault(chunk_key, []).append({
        "id": int(ids[idx]),
        "title": titles[idx],
        "cluster": int(labels[idx]),
        "x": x,
        "y": y
    })

# Save chunks
chunk_metadata = []
for chunk_key, points in spatial_chunks.items():
    grid_x, grid_y = map(int, chunk_key.split('_'))
    chunk_x_min = x_min + grid_x * grid_width
    chunk_x_max = x_min + (grid_x + 1) * grid_width
    chunk_y_min = y_min + grid_y * grid_height
    chunk_y_max = y_min + (grid_y + 1) * grid_height

    filename = f"chunk_{chunk_key}.json"
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        json.dump(points, f)

    chunk_metadata.append({
        "file": filename,
        "bounds": {
            "x_min": float(chunk_x_min),
            "x_max": float(chunk_x_max),
            "y_min": float(chunk_y_min),
            "y_max": float(chunk_y_max)
        },
        "count": len(points)
    })

print(f"Saved {len(spatial_chunks)} spatial chunks")

# Save metadata
metadata = {
    "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
    "grid_size": SPATIAL_GRID_SIZE,
    "total_points": int(len(filtered_indices)),
    "num_clusters": int(n_clusters),
    "clusters": [cluster_info[cid] for cid in sorted(cluster_info.keys()) if cid != -1],
    "chunks": chunk_metadata
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata.json")
print(f"\nDone! Serve the '{OUTPUT_DIR}' directory with a HTTP server.")
