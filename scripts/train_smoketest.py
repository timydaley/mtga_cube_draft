"""End-to-end smoke test for the cloud training loop.

Validates the full plumbing a real training run depends on, in one short
process: card data load -> cube vocab -> CubeDraftEnv rollout with bot
opponents -> a tiny policy net trained on the GPU to imitate the raredraft
heuristic -> Weights & Biases logging -> checkpoint write. It is NOT real
training; it exists so `cloud_bootstrap.sh` can prove a fresh pod is wired
up correctly before you launch anything expensive.

Usage:
    python scripts/train_smoketest.py --wandb-mode online --steps 5
    python scripts/train_smoketest.py --wandb-mode disabled   # offline plumbing check

A clean run prints `SMOKETEST PASSED` as its final line and exits 0. Any
failure (missing data, no CUDA when required, non-finite loss) exits non-zero
with a traceback, so callers can branch on the exit code.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()  # pull .env into os.environ for local dev (WANDB_API_KEY, etc.)
except ImportError:
    pass  # not installed in the cloud image; env is set from pod secrets

from cube_draft.bots.random_bot import RandomBot  # noqa: E402
from cube_draft.bots.raredraft import RaredraftBot  # noqa: E402
from cube_draft.cards.vocab import CardVocab, CubeVocab  # noqa: E402
from cube_draft.env.cube_draft import CubeDraftEnv, DraftConfig  # noqa: E402

logger = logging.getLogger("train_smoketest")


class TinyPolicy(nn.Module):
    """A minimal pick policy over the cube vocab.

    Input is the concatenation of the pack / pool / seen_unpicked count
    vectors (3 * |cube|); output is one logit per cube card. Illegal picks
    are masked by the caller before the loss/argmax.
    """

    def __init__(self, vocab_size: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * vocab_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_device(requested: str) -> torch.device:
    """Resolve the training device, honoring an explicit request."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def build_cube(cards_path: Path, cube_size: int) -> CubeVocab:
    """Build a synthetic cube from the first `cube_size` cards of the vocab.

    The smoke test doesn't need a *real* cube list — only enough distinct
    cards to fill one pack-round (pack_size * num_seats). We take a stable
    prefix of the global vocab so runs are reproducible.
    """
    global_vocab = CardVocab.from_parquet(cards_path)
    if len(global_vocab) < cube_size:
        cube_size = len(global_vocab)
        logger.warning("vocab has only %d cards; shrinking cube to that", cube_size)
    oracle_ids = [global_vocab.card(i).oracle_id for i in range(cube_size)]
    cube = CubeVocab(global_vocab, oracle_ids)
    logger.info("built cube vocab of %d cards from %s", cube.size, cards_path)
    return cube


def make_env(cube: CubeVocab, seed: int) -> CubeDraftEnv:
    """An 8-seat draft with random opponents, seeded for reproducibility."""
    cfg = DraftConfig(seed=seed)
    opponents = [RandomBot(seed=seed + 1 + i) for i in range(cfg.num_seats - 1)]
    return CubeDraftEnv(cube, opponents, config=cfg)


def obs_to_features(obs: dict) -> np.ndarray:
    """Flatten an env observation into the policy's input vector."""
    return np.concatenate(
        [
            obs["pack"].astype(np.float32),
            obs["pool"].astype(np.float32),
            obs["seen_unpicked"].astype(np.float32),
        ]
    )


def collect_episode(
    env: CubeDraftEnv, teacher: RaredraftBot, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one full draft, following the teacher's picks.

    Returns (features, masks, labels) over every agent decision in the
    episode. The label at each step is the teacher's pick; the agent
    follows it so the trajectory stays on the teacher's distribution.
    """
    obs, _ = env.reset(seed=seed)
    feats: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    terminated = False
    while not terminated:
        pick = teacher.pick(obs)
        feats.append(obs_to_features(obs))
        masks.append((obs["pack"] > 0).astype(np.float32))
        labels.append(pick)
        obs, _reward, terminated, _truncated, _info = env.step(pick)
    return np.stack(feats), np.stack(masks), np.array(labels, dtype=np.int64)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    logger.info("device: %s", device)
    if device.type == "cuda":
        logger.info("gpu: %s", torch.cuda.get_device_name(0))

    cube = build_cube(args.data, args.cube_size)
    env = make_env(cube, seed=args.seed)
    teacher = RaredraftBot(cube, seed=args.seed)

    model = TinyPolicy(cube.size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        job_type="smoketest",
        config={
            "steps": args.steps,
            "cube_size": cube.size,
            "lr": args.lr,
            "device": device.type,
            "params": sum(p.numel() for p in model.parameters()),
        },
    )

    last_loss = float("nan")
    for step in range(args.steps):
        feats, masks, labels = collect_episode(env, teacher, seed=args.seed + step)
        x = torch.from_numpy(feats).to(device)
        mask = torch.from_numpy(masks).to(device)
        y = torch.from_numpy(labels).to(device)

        logits = model(x)
        # Mask illegal picks before the softmax so the loss only ranks
        # cards actually in the pack.
        logits = logits.masked_fill(mask == 0, -1e9)
        loss = F.cross_entropy(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == y).float().mean().item()
        last_loss = loss.item()
        logger.info(
            "step %d/%d  loss=%.4f  imitation_acc=%.3f  (%d decisions)",
            step + 1,
            args.steps,
            last_loss,
            acc,
            len(labels),
        )
        wandb.log({"step": step, "loss": last_loss, "imitation_acc": acc})

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cube_size": cube.size, "steps": args.steps},
        args.checkpoint,
    )
    logger.info("wrote checkpoint -> %s", args.checkpoint)
    wandb.summary["final_loss"] = last_loss
    run.finish()

    # Final, greppable success line. Keep this as the last stdout output.
    print(
        f"SMOKETEST PASSED  device={device.type}  cube={cube.size}  "
        f"steps={args.steps}  final_loss={last_loss:.4f}  checkpoint={args.checkpoint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5, help="number of gradient steps (episodes)")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/cards.parquet"),
        help="Scryfall card Parquet (pulled by cloud_bootstrap.sh)",
    )
    parser.add_argument("--cube-size", type=int, default=360, help="cards in the synthetic cube")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="training device"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/smoketest.pt"),
        help="where to write the checkpoint",
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    parser.add_argument("--wandb-project", default="cube-draft", help="W&B project name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.data.exists():
        raise FileNotFoundError(
            f"card data not found at {args.data}. Run scripts/download_scryfall.py "
            f"or pull it from R2 (scripts/r2_sync.py pull scryfall/oracle_cards.parquet {args.data})."
        )

    train(args)


if __name__ == "__main__":
    main()
