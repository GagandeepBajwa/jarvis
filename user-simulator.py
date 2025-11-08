#!/usr/bin/env python3
# pubsub_publisher_static.py
# Publishes native JSONL events to a real Pub/Sub topic (no CLI args).

import json
from typing import Dict, Iterator

# ========= STATIC CONFIG (edit these) =========
PROJECT_ID = "us-con-gcp-sbx-0001192-100925"   # your GCP project
TOPIC_NAME = "ca-hk-team2-jrvs-events"                   # your existing topic
INPUT_FILE = "test-data.json"             # decoded events (one JSON per line)

# Optional: set to a non-empty string to use Pub/Sub ordering keys (must be enabled on topic)
ORDERING_KEY = None  # e.g., "events-stream-1"
# ==============================================

# Lazy import so the file loads even if package missing
try:
    from google.cloud import pubsub_v1
except Exception:
    pubsub_v1 = None


def read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as ex:
                print(f"[WARN] Skipping invalid JSON on line {i}: {ex}")


def publish_events() -> int:
    if pubsub_v1 is None:
        raise RuntimeError("Missing dependency: google-cloud-pubsub. Install with: pip install google-cloud-pubsub")

    events = list(read_jsonl(INPUT_FILE))
    if not events:
        print(f"[ERROR] No valid events found in {INPUT_FILE}")
        return 1

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

    print(f"[INFO] Project={PROJECT_ID} Topic={TOPIC_NAME}  Events={len(events)}")
    sent = 0
    for idx, payload in enumerate(events, 1):
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        # You can also add attributes if you like (e.g., source) — your push handler ignores them today.
        future = publisher.publish(topic_path, data=data)
        msg_id = future.result(timeout=30)
        sent += 1
        print(f"[{idx}/{len(events)}] published msg_id={msg_id} source={payload.get('source','?')}")

    print(f"[OK] Published {sent} event(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(publish_events())
