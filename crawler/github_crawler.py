"""
Crawl top GitHub repos by stars and save features to CSV.

Usage:
    export GITHUB_TOKEN=ghp_xxx
    python github_crawler.py
    NOTE: replace ghp_xxx with github token
    also pip install requests
"""

import csv
import os
import time
import requests

TOKEN = os.environ.get("GITHUB_TOKEN")
TARGET = 2000
MIN_STARS = 50
OUTPUT = "repos.csv"

HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"


def get(url, params=None):
    """GET with simple rate-limit handling."""
    while True:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            wait = max(int(r.headers.get("X-RateLimit-Reset", 0)) - int(time.time()), 1) + 1
            print(f"Rate limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()


def search_repos():
    """Yield up to TARGET repos sorted by stars descending."""
    for page in range(1, 11):  # 10 pages * 100 = 1000 max
        data = get("https://api.github.com/search/repositories", {
            "q": f"stars:>={MIN_STARS}",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page,
        })
        for item in data.get("items", []):
            yield item
        time.sleep(2)  # stay under 30 req/min


def main():
    fields = [
        "full_name", "stargazers_count", "forks_count", "subscribers_count",
        "open_issues_count", "size", "network_count", "language",
        "has_wiki", "has_pages", "has_issues", "topics_count", "age_days",
        "created_at", "pushed_at",
    ]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, hit in enumerate(search_repos(), 1):
            if i > TARGET:
                break
            try:
                d = get(f"https://api.github.com/repos/{hit['full_name']}")
                created = d["created_at"]
                age_days = (time.time() - time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ"))) / 86400
                writer.writerow({
                    "full_name": d["full_name"],
                    "stargazers_count": d["stargazers_count"],
                    "forks_count": d["forks_count"],
                    "subscribers_count": d["subscribers_count"],
                    "open_issues_count": d["open_issues_count"],
                    "size": d["size"],
                    "network_count": d.get("network_count", 0),
                    "language": d.get("language") or "",
                    "has_wiki": d["has_wiki"],
                    "has_pages": d["has_pages"],
                    "has_issues": d["has_issues"],
                    "topics_count": len(d.get("topics") or []),
                    "age_days": int(age_days),
                    "created_at": d["created_at"],
                    "pushed_at": d["pushed_at"],
                })
                print(f"[{i}/{TARGET}] {d['full_name']} stars={d['stargazers_count']}")
            except Exception as e:
                print(f"Failed on {hit['full_name']}: {e}")


if __name__ == "__main__":
    main()
