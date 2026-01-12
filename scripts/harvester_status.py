#!/usr/bin/env python3
"""Harvester status diagnostics."""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(".local/tools/browser-automation/fetch_log.jsonl")
INDEX_FILE = Path(".local/data/scans/mindmap_index.json")


def get_session_stats():
    """Get stats from most recent session."""
    if not LOG_FILE.exists():
        print("No fetch log found")
        return

    sessions = []
    current_session = {"fetches": [], "start": None, "end": None}

    with open(LOG_FILE) as f:
        for line in f:
            try:
                d = json.loads(line)
                event = d.get("event")
                if event == "session_start":
                    current_session = {"fetches": [], "start": d, "end": None}
                elif event == "session_end":
                    current_session["end"] = d
                    sessions.append(current_session)
                elif event == "fetch":
                    current_session["fetches"].append(d)
            except:
                pass

    # If no end, session may still be running
    if current_session["start"] and not current_session["end"]:
        sessions.append(current_session)

    if not sessions:
        print("No sessions found in log")
        return

    last = sessions[-1]
    print("=== Latest Session ===")
    if last["start"]:
        start_time = last["start"]["timestamp"]
        runtime = last["start"].get("max_runtime_hours", "?")
        remaining = last["start"].get("remaining_trees", "?")
        print(f"Started: {start_time}")
        print(f"Max runtime: {runtime}h")
        print(f"Trees in queue: {remaining}")

    if last["end"]:
        print(f"Ended: {last['end']['timestamp']}")
        print(f"Fetched: {last['end'].get('total_fetched', '?')}")
        print(f"Errors: {last['end'].get('total_errors', '?')}")
        print(f"Remaining: {last['end'].get('remaining_trees', '?')}")
    else:
        print(f"Status: RUNNING (or incomplete)")
        print(f"Fetches so far: {len(last['fetches'])}")

    if last["fetches"]:
        print(f"\nLast fetch: {last['fetches'][-1].get('title', '?')}")
        print(f"  at {last['fetches'][-1].get('timestamp', '?')}")

    # Account distribution in this session
    accounts = Counter()
    for f in last["fetches"]:
        # Would need to cross-reference with targets to get account
        pass


def get_queue_stats(targets_file: str = None):
    """Get queue statistics."""
    if targets_file is None:
        # Try common targets files
        candidates = [
            "reports/pearltrees_targets_combined_2026-01-02_trees.jsonl",
            "reports/pearltrees_targets_s243a.jsonl",
        ]
        for c in candidates:
            if Path(c).exists():
                targets_file = c
                break

    if not targets_file or not Path(targets_file).exists():
        print("No targets file found")
        return

    # Load index to find processed trees
    processed = set()
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            processed = set(json.load(f).keys())

    # Count by account
    accounts = Counter()
    total = 0
    remaining = 0

    with open(targets_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                total += 1
                acct = d.get("account", "unknown")
                tree_id = d.get("tree_id")
                if tree_id not in processed:
                    remaining += 1
                    accounts[acct] += 1
            except:
                pass

    print(f"\n=== Queue: {targets_file} ===")
    print(f"Total targets: {total}")
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {remaining}")

    print("\nRemaining by account:")
    for acct, count in accounts.most_common(10):
        pct = 100 * count / remaining if remaining else 0
        print(f"  {acct}: {count} ({pct:.1f}%)")


def main():
    targets = sys.argv[1] if len(sys.argv) > 1 else None
    get_session_stats()
    get_queue_stats(targets)


if __name__ == "__main__":
    main()
