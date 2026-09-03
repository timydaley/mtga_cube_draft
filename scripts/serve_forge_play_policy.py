#!/usr/bin/env python3
"""Serve a learned Forge play policy over stdin/stdout JSONL.

Protocol:
  input line:  {"request_id": "optional", "state": {...}, "legal_actions": [...]}
  output line: {"request_id": "optional", "action_id": "...", "action_index": 0, "score": ...}

A Java/Forge adapter can keep this process open and query it whenever Forge asks
for an AI decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cube_draft.model.forge_play_policy import ForgePlayPolicy, choose_action
from cube_draft.utils import checkpoint as ckpt
from cube_draft.utils.torch_utils import select_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/forge_play_policy"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "mlx", "cpu"])
    args = p.parse_args()

    device = select_device(args.device)
    state = ckpt.load_checkpoint(args.checkpoint, args.ckpt_dir, device=str(device))
    model = ForgePlayPolicy(**state.get("model_args", {})).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = choose_action(model, req, device)
            if "request_id" in req:
                resp["request_id"] = req["request_id"]
        except Exception as exc:  # keep long-running server alive for adapter debugging
            resp = {"error": str(exc)}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
