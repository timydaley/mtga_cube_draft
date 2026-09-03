#!/usr/bin/env python3
"""Write a tiny synthetic Forge play-policy JSONL log for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/forge_play_policy_smoke/decisions.jsonl"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "state": {"phase": "main1", "life": {"self": 20, "opp": 20}, "hand": ["Mountain", "Lightning Bolt"]},
            "legal_actions": [
                {"id": "pass", "type": "pass", "text": "Pass priority"},
                {"id": "cast_bolt_face", "type": "cast", "card": "Lightning Bolt", "target": "opponent"},
            ],
            "chosen_action_id": "cast_bolt_face",
            "reward": 1.0,
        },
        {
            "state": {"phase": "combat", "life": {"self": 20, "opp": 3}, "battlefield": ["Goblin Guide"]},
            "legal_actions": [
                {"id": "attack_guide", "type": "attack", "card": "Goblin Guide"},
                {"id": "no_attack", "type": "attack", "text": "Attack with none"},
            ],
            "chosen_action_id": "attack_guide",
            "reward": 1.0,
        },
    ]
    with args.out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
