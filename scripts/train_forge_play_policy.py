#!/usr/bin/env python3
"""Train the learned Forge play policy from action logs.

This is the Python-side trainer for the "actual play policy" path.  It expects
JSONL records emitted by a Forge adapter, one decision per line:

    {"state": {...}, "legal_actions": [{"id": "a", ...}, ...], "chosen_action_id": "a", "reward": 1.0}

Initially this supports behavior cloning / weighted behavior cloning from Forge
AI, heuristic, or future model-vs-model logs.  If logs contain per-decision
`reward` or `weight`, winning-game actions can be upweighted.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:
    pass

from cube_draft.model.forge_play_policy import ForgePlayPolicy, parse_play_example, hashed_features
from cube_draft.utils import checkpoint as ckpt
from cube_draft.utils.torch_utils import select_device

logger = logging.getLogger("train_forge_play_policy")


def load_examples(paths: list[Path]) -> list:
    examples = []
    skipped = 0
    for path in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("bad json %s:%d", path, line_no)
                    skipped += 1
                    continue
                ex = parse_play_example(rec)
                if ex is None:
                    skipped += 1
                else:
                    examples.append(ex)
    logger.info("loaded %d examples skipped=%d", len(examples), skipped)
    return examples


def make_batch(examples, idx: np.ndarray, state_dim: int, action_dim: int, device: torch.device):
    batch = [examples[int(i)] for i in idx]
    max_actions = max(len(ex.legal_actions) for ex in batch)
    state = np.zeros((len(batch), state_dim), dtype=np.float32)
    actions = np.zeros((len(batch), max_actions, action_dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_actions), dtype=bool)
    chosen = np.zeros(len(batch), dtype=np.int64)
    weight = np.zeros(len(batch), dtype=np.float32)

    for row, ex in enumerate(batch):
        state[row] = hashed_features(ex.state, state_dim, "state")
        for j, action in enumerate(ex.legal_actions):
            actions[row, j] = hashed_features(action, action_dim, "action")
            mask[row, j] = True
        chosen[row] = ex.chosen_index
        weight[row] = max(0.0, float(ex.reward))

    return (
        torch.as_tensor(state, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
        torch.as_tensor(chosen, dtype=torch.long, device=device),
        torch.as_tensor(weight, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def evaluate(model, examples, batch_size: int, device: torch.device) -> dict[str, float]:
    if not examples:
        return {"acc": 0.0, "loss": 0.0}
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for start in range(0, len(examples), batch_size):
        idx = np.arange(start, min(len(examples), start + batch_size))
        state, actions, mask, chosen, weight = make_batch(examples, idx, model.state_dim, model.action_dim, device)
        logits = model(state, actions, mask)
        loss = F.cross_entropy(logits, chosen, reduction="none")
        weight = weight.clamp_min(1e-6)
        loss_sum += float((loss * weight).sum().item())
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == chosen).sum().item())
        total += len(idx)
    return {"acc": correct / max(1, total), "loss": loss_sum / max(1, total)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, nargs="+", required=True, help="Forge decision JSONL logs")
    p.add_argument("--resume", default=None, help="checkpoint path/R2 key/latest")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/forge_play_policy"))
    p.add_argument("--state-dim", type=int, default=4096)
    p.add_argument("--action-dim", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "mlx", "cpu"])
    p.add_argument("--r2-prefix", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    examples = load_examples(args.logs)
    if not examples:
        raise ValueError("no trainable examples found")
    order = np.arange(len(examples))
    rng.shuffle(order)
    n_eval = int(round(len(order) * args.eval_frac))
    eval_idx = order[:n_eval]
    train_idx = order[n_eval:]
    train = [examples[int(i)] for i in train_idx]
    eval_ = [examples[int(i)] for i in eval_idx]
    logger.info("split train=%d eval=%d device=%s", len(train), len(eval_), device)

    if args.resume:
        state = ckpt.load_checkpoint(args.resume, args.ckpt_dir, device=str(device))
        model = ForgePlayPolicy(**state.get("model_args", {})).to(device)
        model.load_state_dict(state["model"])
    else:
        model = ForgePlayPolicy(args.state_dim, args.action_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_order = np.arange(len(train))
        rng.shuffle(train_order)
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(train_order), args.batch_size):
            idx = train_order[start : start + args.batch_size]
            state_t, actions_t, mask_t, chosen_t, weight_t = make_batch(
                train, idx, model.state_dim, model.action_dim, device
            )
            logits = model(state_t, actions_t, mask_t)
            losses = F.cross_entropy(logits, chosen_t, reduction="none")
            weight_t = weight_t.clamp_min(1e-6)
            loss = (losses * weight_t).sum() / weight_t.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            total_loss += float(loss.item()) * len(idx)
            total_n += len(idx)
        metrics = evaluate(model, eval_, args.batch_size, device) if eval_ else {"acc": 0.0, "loss": 0.0}
        logger.info(
            "epoch=%d train_loss=%.4f eval_loss=%.4f eval_acc=%.4f",
            epoch,
            total_loss / max(1, total_n),
            metrics["loss"],
            metrics["acc"],
        )
        ckpt.save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "model_args": {"state_dim": model.state_dim, "action_dim": model.action_dim, "hidden": args.hidden},
                "args": vars(args),
            },
            args.ckpt_dir,
            step,
            r2_prefix=args.r2_prefix,
        )

    print(f"FORGE PLAY POLICY DONE ckpt={args.ckpt_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
