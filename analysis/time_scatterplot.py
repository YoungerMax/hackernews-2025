import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Load your dataset
# ----------------------------
df = pd.read_csv("hackernews.csv")

# Drop rows missing essentials
df = df.dropna(subset=["score", "time_unix"])

# ----------------------------
# 2. Create time features
# ----------------------------
df["datetime"] = pd.to_datetime(df["time_unix"], unit="s", utc=True)
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday  # 0=Mon, 6=Sun

# ----------------------------
# 3. Create binary high-score label
# ----------------------------
df["high_score"] = (df["score"] >= 200).astype(int)

# ----------------------------
# 4. Prepare heatmap data
# ----------------------------
# Aggregate counts of posts by weekday and hour
heatmap_data = df.groupby(["weekday", "hour", "high_score"]).size().reset_index(name="count")

# Pivot for heatmap: high_score=1 and high_score=0 separately
heatmap_high = heatmap_data[heatmap_data["high_score"]==1].pivot(index="hour", columns="weekday", values="count").fillna(0)
heatmap_other = heatmap_data[heatmap_data["high_score"]==0].pivot(index="hour", columns="weekday", values="count").fillna(0)

# ----------------------------
# 5. Plot heatmaps
# ----------------------------
sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# High-score posts
sns.heatmap(
    heatmap_high,
    ax=axes[0],
    cmap="Reds",
    cbar_kws={'label': 'Number of posts'},
    linewidths=0.5
)
axes[0].set_title("Posts with 200+ Points")
axes[0].set_xlabel("Weekday (0=Mon)")
axes[0].set_ylabel("Hour of Day")

# Other posts
sns.heatmap(
    heatmap_other,
    ax=axes[1],
    cmap="Blues",
    cbar_kws={'label': 'Number of posts'},
    linewidths=0.5
)
axes[1].set_title("Posts with <200 Points")
axes[1].set_xlabel("Weekday (0=Mon)")
axes[1].set_ylabel("Hour of Day")

plt.tight_layout()
plt.savefig("hn_post_heatmap.png", dpi=150)
print("Saved heatmap as hn_post_heatmap.png")

