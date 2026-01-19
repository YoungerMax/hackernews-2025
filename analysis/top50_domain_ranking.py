import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract

bar = 0

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("hackernews.csv")

# Drop rows missing URL or score
df = df.dropna(subset=["url", "score"])

# ----------------------------
# 2. Extract domain
# ----------------------------
def extract_domain(url):
    ext = tldextract.extract(url)
    return ext.domain if ext.domain else "none"

df["domain"] = df["url"].apply(extract_domain)

# ----------------------------
# 3. Filter high-score posts
# ----------------------------
high_score_df = df[df["score"] >= bar]

# ----------------------------
# 4. Aggregate by domain
# ----------------------------
domain_counts = (
    high_score_df.groupby("domain")
    .size()
    .reset_index(name="high_score_posts")
    .sort_values(by="high_score_posts", ascending=False)
)

# Keep only top 50 domains
top_domains = domain_counts.head(50)

# ----------------------------
# 5. Plot horizontal bar chart
# ----------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))

sns.barplot(
    x="high_score_posts",
    y="domain",
    data=top_domains,
    palette="viridis"
)

plt.title(f"Top 50 domains of Hacker News posts with at least {bar} points")
plt.xlabel("Number of Posts")
plt.ylabel("Domain")
plt.tight_layout()

# Save figure
plt.savefig("hn_top50_domains.png", dpi=150)
print("Saved top 50 domain ranking as 'hn_top50_domains.png'")
