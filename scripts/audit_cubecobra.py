"""Week-0 gating audit (plan §4.1).

Question: in the CubeCobra S3 export, are picks flagged as human-seat
vs. bot-seat? If not, all Stage 2 training is implicitly distilling the
CubeCobra bot — drives the entire framing of §4.7.

This script:
1. Lists the export prefix and prints what's available.
2. Downloads a small sample of picks/*.json and cubeInstances/*.json.
3. Pretty-prints the top-level schema and reports whether any seat-type
   or agent flag is present per pick.
4. Counts unique "seat" / "agent" / "user_id" keys, if found.

Output is a human-readable report. Treat as the source for the Discord
question to @dekkerglen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cube_draft.data.cubecobra import (
    fetch_cube_instances_sample,
    fetch_picks_sample,
    list_export_objects,
)

SEAT_KEY_CANDIDATES = ("seat", "seatIndex", "seat_idx", "agent", "agentType", "isHuman", "userId", "user")


def _walk_keys(obj: Any, into: Counter, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            into[path] += 1
            _walk_keys(v, into, path)
    elif isinstance(obj, list) and obj:
        # Walk only the first element; same schema repeats.
        _walk_keys(obj[0], into, prefix + "[]")


def summarize_picks(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    keys = Counter()
    _walk_keys(data, keys)
    seat_hits = {k: keys[k] for k in keys if any(c.lower() in k.lower() for c in SEAT_KEY_CANDIDATES)}
    sample = data if isinstance(data, dict) else (data[:1] if isinstance(data, list) else data)
    return {
        "top_level_type": type(data).__name__,
        "top_level_len": len(data) if hasattr(data, "__len__") else None,
        "all_keys": dict(keys.most_common(40)),
        "seat_key_candidates_found": seat_hits,
        "sample_entry": sample,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/cubecobra_audit"))
    p.add_argument("--n-samples", type=int, default=3)
    args = p.parse_args()

    print("=== CubeCobra export — top of prefix ===")
    objs = list_export_objects(max_keys=30)
    for o in objs:
        print(f"  {o['Key']}  ({o['Size']} bytes)")
    if not objs:
        print("  (no objects returned — bucket layout may have changed)")
        return

    print()
    print(f"=== Downloading {args.n_samples} picks/*.json samples ===")
    picks_paths = fetch_picks_sample(args.out_dir / "picks", n=args.n_samples)
    for path in picks_paths:
        print(f"\n--- {path.name} ---")
        summary = summarize_picks(path)
        for k, v in summary.items():
            if k == "sample_entry":
                print(f"{k}: <see file>")
            else:
                print(f"{k}: {v}")

    print()
    print(f"=== Downloading {args.n_samples} cubeInstances/*.json samples ===")
    ci_paths = fetch_cube_instances_sample(args.out_dir / "cubeInstances", n=args.n_samples)
    for path in ci_paths:
        print(f"\n--- {path.name} ---")
        summary = summarize_picks(path)
        for k, v in summary.items():
            if k == "sample_entry":
                print(f"{k}: <see file>")
            else:
                print(f"{k}: {v}")

    print()
    print("=== AUDIT DECISION ===")
    print("Inspect 'seat_key_candidates_found' above. If any pick-level key resembles")
    print("a seat/agent/user identifier with consistent non-trivial values, we can")
    print("filter to human picks. If none, proceed with the distillation framing")
    print("from plan §4.1 step 4.")


if __name__ == "__main__":
    main()
