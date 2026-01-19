import asyncio
import aiohttp
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta

BASE_URL = "https://hacker-news.firebaseio.com/v0"
OUTPUT_FILE = "hackernews.csv"
CHECKPOINT_FILE = "hn_checkpoint.json"


CUTOFF_TS = datetime(2025, 1, 1).timestamp()

CONCURRENCY = 40
BATCH_SIZE = 200
FLUSH_EVERY = 50

last_item_id = None
request_count = 0
start_time = time.time()


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f).get("last_item_id")
    return None


def save_checkpoint(item_id):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_item_id": item_id}, f)


def print_meter():
    elapsed = time.time() - start_time
    if elapsed > 0:
        rps = request_count / elapsed
        print(f"\rRequests: {request_count} | {rps:6.1f} req/s", end="", flush=True)


async def fetch_item(session, semaphore, item_id):
    global request_count

    async with semaphore:
        try:
            async with session.get(f"{BASE_URL}/item/{item_id}.json") as resp:
                request_count += 1
                if resp.status == 200:
                    return await resp.json()
        except aiohttp.ClientError:
            pass
    return None


async def process_batch(session, semaphore, writer, f, batch_ids):
    global last_item_id

    tasks = [fetch_item(session, semaphore, i) for i in batch_ids]
    
    # Collect all results first
    results = []
    for coro in asyncio.as_completed(tasks):
        item = await coro
        if item and "time" in item:
            results.append(item)
    
    # Sort by ID to ensure we process in order
    results.sort(key=lambda x: x["id"], reverse=True)
    
    # Now process in order
    cutoff_reached = False
    for item in results:
        last_item_id = item["id"]

        if item["time"] < CUTOFF_TS:
            cutoff_reached = True
            save_checkpoint(last_item_id)
            break  # Stop processing, but all valid items before this are saved

        if item.get("type") != "story":
            continue

        writer.writerow([
            item.get("id"),
            item.get("type"),
            item.get("by"),
            item.get("time"),
            datetime.utcfromtimestamp(item["time"]).isoformat(),
            item.get("title", ""),
            item.get("score", 0),
            item.get("descendants", 0),
            item.get("url", ""),
            item.get("text", "")
        ])

        if item["id"] % FLUSH_EVERY == 0:
            f.flush()
            save_checkpoint(item["id"])
    
    # Return whether we hit cutoff
    return cutoff_reached


async def main():
    global last_item_id

    checkpoint = load_checkpoint()

    async with aiohttp.ClientSession(
        headers={"User-Agent": "hn-async-archive/3.0"}
    ) as session:

        async with session.get(f"{BASE_URL}/maxitem.json") as resp:
            max_item = await resp.json()

        start_id = checkpoint if checkpoint else max_item
        mode = "a" if os.path.exists(OUTPUT_FILE) else "w"

        print(f"Starting from item id: {start_id}")

        semaphore = asyncio.Semaphore(CONCURRENCY)

        with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if mode == "w":
                writer.writerow([
                    "id",
                    "type",
                    "by",
                    "time_unix",
                    "time_iso",
                    "title",
                    "score",
                    "descendants",
                    "url",
                    "text"
                ])
                f.flush()

            for batch_start in range(start_id, 0, -BATCH_SIZE):
                batch_ids = range(
                    batch_start,
                    max(batch_start - BATCH_SIZE, 0),
                    -1
                )

                cutoff_reached = await process_batch(
                    session, semaphore, writer, f, batch_ids
                )

                save_checkpoint(batch_start)
                print_meter()
                
                if cutoff_reached:
                    print("\nCutoff reached. Stopping.")
                    break

    save_checkpoint(last_item_id)
    print("\nFinished cleanly.")


def handle_sigint(signum, frame):
    print("\nCtrl+C detected. Saving checkpoint...")
    if last_item_id:
        save_checkpoint(last_item_id)
        print(f"Checkpoint saved at item {last_item_id}")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    asyncio.run(main())