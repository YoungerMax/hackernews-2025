#!/usr/bin/env python3

import csv
import tempfile
import urllib.request
from urllib.parse import urlparse
import os
import numpy as np

CSV_FILE = "hackernews.csv"

BLACKLIST_URLS = [
    "https://malware-filter.gitlab.io/malware-filter/phishing-filter-dnscrypt-blocked-names.txt",
    "https://malware-filter.gitlab.io/malware-filter/urlhaus-filter-dnscrypt-blocked-names.txt",
    "https://curbengh.github.io/pup-filter/pup-filter-dnscrypt-blocked-names.txt",
    "https://curbengh.github.io/vn-badsite-filter/vn-badsite-filter-dnscrypt-blocked-names.txt"
]


def download_blacklists():
    """
    Download all blacklists to temporary files.
    Returns list of file paths.
    """
    paths = []

    for url in BLACKLIST_URLS:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        with urllib.request.urlopen(url) as resp, open(tmp.name, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        paths.append(tmp.name)

    return paths


def load_blacklists(paths):
    """
    Load all blacklist domains into a single set.
    """
    blacklist = set()

    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().lower()
                if not line or line.startswith("#"):
                    continue
                blacklist.add(line)

    return blacklist


def extract_domain(url):
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return None


def is_blacklisted(domain, blacklist):
    """
    Check domain and all parent domains.
    """
    if not domain:
        return False

    parts = domain.split(".")
    for i in range(len(parts)):
        if ".".join(parts[i:]) in blacklist:
            return True
    return False


def five_number_summary(values):
    if not values:
        return None

    arr = np.array(values)
    return {
        "min": int(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q3": float(np.percentile(arr, 75)),
        "max": int(np.max(arr)),
    }


def main():
    print("Downloading blacklists...")
    blacklist_paths = download_blacklists()

    try:
        print("Loading blacklists...")
        blacklist = load_blacklists(blacklist_paths)
        print(f"Loaded {len(blacklist):,} unique blacklisted domains\n")

        total_posts = 0
        blacklisted_posts = 0

        scores_blacklisted = []
        scores_non_blacklisted = []

        print("Processing hackernews.csv...")
        with open(CSV_FILE, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_posts += 1

                # score
                try:
                    score = int(row.get("score"))
                except Exception:
                    score = None

                # url/domain
                url = row.get("url")
                domain = extract_domain(url) if url else None

                if is_blacklisted(domain, blacklist):
                    blacklisted_posts += 1
                    if score is not None:
                        scores_blacklisted.append(score)
                else:
                    if score is not None:
                        scores_non_blacklisted.append(score)

        pct_blacklisted = (blacklisted_posts / total_posts) * 100 if total_posts else 0

        summary_blacklisted = five_number_summary(scores_blacklisted)
        summary_non_blacklisted = five_number_summary(scores_non_blacklisted)

        print("\n===== SUMMARY =====")
        print(f"Total posts: {total_posts:,}")
        print(f"Blacklisted posts: {blacklisted_posts:,} ({pct_blacklisted:.2f}%)")
        print(
            f"Non-blacklisted posts: {total_posts - blacklisted_posts:,} "
            f"({100 - pct_blacklisted:.2f}%)"
        )

        print("\n===== SCORE DISTRIBUTION (5-NUMBER SUMMARY) =====")

        print("\nBlacklisted domains:")
        if summary_blacklisted:
            for k, v in summary_blacklisted.items():
                print(f"  {k:>6}: {v}")
        else:
            print("  No score data")

        print("\nNon-blacklisted domains:")
        if summary_non_blacklisted:
            for k, v in summary_non_blacklisted.items():
                print(f"  {k:>6}: {v}")
        else:
            print("  No score data")

    finally:
        for path in blacklist_paths:
            os.remove(path)


if __name__ == "__main__":
    main()
