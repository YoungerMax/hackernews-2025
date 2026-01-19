import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("hackernews.csv")

# Drop rows with missing titles
df = df.dropna(subset=["title"])

# ----------------------------
# 2. Clean titles
# ----------------------------
def clean_title(title):
    title = title.lower()
    
    # Remove common HN boilerplate
    boilerplate_patterns = [
        r"^ask hn[:\-]?",     # "Ask HN"
        r"^show hn[:\-]?",    # "Show HN"
        r"is hiring",         # "is hiring"
        r"launch hn",         # "launch"
        r"pdf",           # PDF links
        r"video",         # Video links
    ]
    
    for pattern in boilerplate_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    # Remove non-alphanumeric characters
    title = re.sub(r"[^a-z0-9\s]", ' ', title)
    
    # Collapse multiple spaces
    title = re.sub(r"\s+", ' ', title).strip()
    
    return title

df["clean_title"] = df["title"].apply(clean_title)

# ----------------------------
# 3. TF-IDF vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,1),  # unigrams
    max_features=50000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_title"])

# ----------------------------
# 4. Aggregate TF-IDF scores
# ----------------------------
# Sum TF-IDF scores across all documents to get keyword importance
import numpy as np

tfidf_sum = np.array(X.sum(axis=0)).flatten()
tfidf_scores = pd.DataFrame({
    'term': vectorizer.get_feature_names_out(),
    'tfidf': tfidf_sum
})

# ----------------------------
# 5. Rank top 50 keywords
# ----------------------------
top_keywords = tfidf_scores.sort_values(by="tfidf", ascending=False).head(50)
print(top_keywords)

# ----------------------------
# 6. Save to CSV (optional)
# ----------------------------
top_keywords.to_csv("hn_top50_keywords.csv", index=False)
print("Saved top 50 keywords as 'hn_top50_keywords.csv'")
