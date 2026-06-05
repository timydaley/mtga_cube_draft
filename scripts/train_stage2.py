"""Stage 2 — train the CC-CPR model by distilling a teacher bot (plan §4.4/§4.5).

Per the Week 0 audit (AUDIT_WEEK0.md) Stage 2 is framed as distillation: learn a
small fast model that matches a teacher drafter. This entrypoint distills a
*local* teacher (RaredraftBot by default) over drafts generated on the fly by
the simulator, so it runs end-to-end on a GPU pod with no external data ETL.
Swapping in the CubeCobra-export dataset later only replaces the data-generation
step; the model, loss, eval, and checkpointing are unchanged.

What it does:
  - builds N random cubes from the Scryfall vocab (generalization signal),
  - generates teacher drafts on each (all 8 seats = teacher), recording every
    pick decision (pack / pool / seen_unpicked / pick context -> picked card),
  - trains CC-CPR with the §4.4 triplet loss, annealing the feature-mix alpha,
  - evaluates top-1 / top-3 / MRR pick agreement with the teacher on held-out
    drafts, against the random-pick baseline,
  - logs to W&B and checkpoints to R2 (resumable across spot interruptions).

Usage:
    python scripts/train_stage2.py --steps 5000 --r2-prefix checkpoints/stage2/run1
    python scripts/train_stage2.py --resume latest --steps 10000   # continue
    python scripts/train_stage2.py --steps 50 --device cpu --wandb-mode disabled  # local
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:
    pass

from cube_draft.bots.base import Drafter  # noqa: E402
from cube_draft.bots.random_bot import RandomBot  # noqa: E402
from cube_draft.bots.raredraft import RaredraftBot  # noqa: E402
from cube_draft.cards import features  # noqa: E402
from cube_draft.cards.vocab import CardVocab, CubeVocab  # noqa: E402
from cube_draft.env.cube_draft import DraftConfig, SeatState  # noqa: E402
from cube_draft.env.pack import Pack  # noqa: E402
from cube_draft.model.cc_cpr import CCCPR, CubeTensors, DraftBatch, triplet_loss  # noqa: E402
from cube_draft.utils import checkpoint as ckpt  # noqa: E402
from cube_draft.utils.torch_utils import select_device  # noqa: E402

logger = logging.getLogger("train_stage2")


# --------------------------------------------------------------------------
# Cube construction
# --------------------------------------------------------------------------


def build_cube(global_vocab: CardVocab, size: int, rng: np.random.Generator) -> CubeVocab:
    """Sample `size` distinct cards from the global vocab into a cube."""
    size = min(size, len(global_vocab))
    idx = rng.choice(len(global_vocab), size=size, replace=False)
    oracle_ids = [global_vocab.card(int(i)).oracle_id for i in idx]
    return CubeVocab(global_vocab, oracle_ids)


def cube_tensors(cube: CubeVocab) -> CubeTensors:
    cards = [cube.card(i) for i in range(cube.size)]
    return CubeTensors(
        feats=torch.from_numpy(features.feature_matrix(cards)),
        global_idx=torch.tensor(cube.global_indices(), dtype=torch.long),
        colors=torch.from_numpy(features.color_matrix(cards)),
    )


def make_teacher(name: str, cube: CubeVocab, seed: int) -> Drafter:
    if name == "raredraft":
        return RaredraftBot(cube, seed=seed)
    if name == "random":
        return RandomBot(seed=seed)
    raise ValueError(f"unknown teacher: {name}")


# --------------------------------------------------------------------------
# Draft generation (records every seat's pick decision)
# --------------------------------------------------------------------------


def _observe(seat: SeatState, pack: Pack, pack_idx: int, pick_idx: int, seat_idx: int, n: int) -> dict:
    pack_vec = np.zeros(n, dtype=np.int8)
    for c in pack:
        pack_vec[c] = 1
    return {
        "pack": pack_vec,
        "pool": seat.pool_counts(n),
        "seen_unpicked": seat.seen_counts(n),
        "cube_mask": np.ones(n, dtype=np.int8),
        "pack_idx": pack_idx,
        "pick_idx": pick_idx,
        "seat_idx": seat_idx,
    }


def collect_decisions(
    cube: CubeVocab, teacher_name: str, n_drafts: int, cfg: DraftConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Run `n_drafts` all-teacher drafts; record every (obs, picked) decision.

    Every seat is the teacher, so each draft yields num_seats * total_picks
    decisions. seen_unpicked accumulates per the §2.4 own-seat-only signal.
    """
    n = cube.size
    drafters = [
        make_teacher(teacher_name, cube, seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(cfg.num_seats)
    ]
    packs_buf: list[np.ndarray] = []
    pools_buf: list[np.ndarray] = []
    seen_buf: list[np.ndarray] = []
    pack_idx_buf: list[int] = []
    pick_idx_buf: list[int] = []
    picked_buf: list[int] = []

    for _ in range(n_drafts):
        seats = [SeatState() for _ in range(cfg.num_seats)]
        for pack_idx in range(cfg.num_packs):
            n_cards = cfg.pack_size * cfg.num_seats
            sampled = rng.choice(n, size=n_cards, replace=False)
            packs = [
                Pack(sampled[i * cfg.pack_size : (i + 1) * cfg.pack_size].tolist())
                for i in range(cfg.num_seats)
            ]
            direction = 1 if pack_idx % 2 == 0 else -1
            for pick_idx in range(cfg.pack_size):
                for seat in range(cfg.num_seats):
                    pack = packs[seat]
                    obs = _observe(seats[seat], pack, pack_idx, pick_idx, seat, n)
                    choice = drafters[seat].pick(obs)
                    packs_buf.append(obs["pack"])
                    pools_buf.append(obs["pool"])
                    seen_buf.append(obs["seen_unpicked"])
                    pack_idx_buf.append(pack_idx)
                    pick_idx_buf.append(pick_idx)
                    picked_buf.append(choice)
                    seats[seat].seen_unpicked.extend(c for c in pack if c != choice)
                    seats[seat].pool.append(choice)
                    pack.pick(choice)
                # rotate
                new_packs = [Pack([])] * cfg.num_seats
                for i, p in enumerate(packs):
                    new_packs[(i + direction) % cfg.num_seats] = p
                packs = new_packs

    return {
        "pack": np.stack(packs_buf),
        "pool": np.stack(pools_buf),
        "seen": np.stack(seen_buf),
        "pack_idx": np.array(pack_idx_buf, dtype=np.int64),
        "pick_idx": np.array(pick_idx_buf, dtype=np.int64),
        "picked": np.array(picked_buf, dtype=np.int64),
    }


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def make_batch(
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    device: torch.device,
    mask_cube: bool,
    rng: np.random.Generator,
) -> tuple[DraftBatch, torch.Tensor]:
    """Slice a minibatch and move it to-device. Returns (batch, picked)."""
    pack = data["pack"][idx]
    pool = data["pool"][idx]
    seen = data["seen"][idx]
    n = pack.shape[1]

    if mask_cube:
        # §2.5 training-time random cube masking: hide a random 0-80% of the
        # cube list, but always keep cards the model can already see (pack/pool).
        keep_frac = rng.uniform(0.2, 1.0, size=(len(idx), 1))
        cube_mask = (rng.random((len(idx), n)) < keep_frac).astype(np.int8)
        cube_mask = np.maximum(cube_mask, (pack > 0) | (pool > 0))
    else:
        cube_mask = np.ones_like(pack)

    def t(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device)

    batch = DraftBatch(
        pack=t(pack, torch.float32),
        pool=t(pool, torch.float32),
        seen=t(seen, torch.float32),
        cube_mask=t(cube_mask, torch.float32),
        pack_idx=t(data["pack_idx"][idx], torch.long),
        pick_idx=t(data["pick_idx"][idx], torch.long),
    )
    picked = t(data["picked"][idx], torch.long)
    return batch, picked


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: CCCPR, cube_t: CubeTensors, data: dict[str, np.ndarray], device: torch.device
) -> dict[str, float]:
    """Top-1 / top-3 / MRR pick agreement with the teacher, over the in-pack set."""
    model.eval()
    pack = torch.as_tensor(data["pack"], dtype=torch.float32, device=device)
    picked = torch.as_tensor(data["picked"], dtype=torch.long, device=device)
    batch = DraftBatch(
        pack=pack,
        pool=torch.as_tensor(data["pool"], dtype=torch.float32, device=device),
        seen=torch.as_tensor(data["seen"], dtype=torch.float32, device=device),
        cube_mask=torch.ones_like(pack),
        pack_idx=torch.as_tensor(data["pack_idx"], dtype=torch.long, device=device),
        pick_idx=torch.as_tensor(data["pick_idx"], dtype=torch.long, device=device),
    )
    scores = model(batch, cube_t)
    masked = scores.masked_fill(pack == 0, float("-inf"))

    rows = torch.arange(scores.shape[0], device=device)
    picked_score = masked[rows, picked].unsqueeze(1)
    # rank = number of in-pack cards scoring strictly higher than the picked one
    rank = (masked > picked_score).sum(dim=1)  # 0 == top-1
    top1 = (rank == 0).float().mean().item()
    top3 = (rank < 3).float().mean().item()
    mrr = (1.0 / (rank.float() + 1.0)).mean().item()

    # random-pick baseline = mean(1 / pack_size_at_pick)
    pack_sizes = pack.sum(dim=1).clamp_min(1.0)
    random_top1 = (1.0 / pack_sizes).mean().item()
    return {"top1": top1, "top3": top3, "mrr": mrr, "random_top1": random_top1}


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    logger.info("device: %s", device)
    if device.type == "cuda":
        logger.info("gpu: %s", torch.cuda.get_device_name(0))

    global_vocab = CardVocab.from_parquet(args.data)
    logger.info("global vocab: %d cards", len(global_vocab))

    cfg = DraftConfig()
    # Build cubes + their tensors + per-cube teacher train/eval decision buffers.
    cube_ts: list[CubeTensors] = []
    train_data: list[dict[str, np.ndarray]] = []
    eval_data: list[dict[str, np.ndarray]] = []
    for c in range(args.num_cubes):
        cube = build_cube(global_vocab, args.cube_size, rng)
        logger.info("cube %d: %d cards; generating %d teacher drafts", c, cube.size, args.drafts_per_cube)
        cube_ts.append(cube_tensors(cube).to(device))
        train_data.append(collect_decisions(cube, args.teacher, args.drafts_per_cube, cfg, rng))
        eval_data.append(collect_decisions(cube, args.teacher, args.eval_drafts, cfg, rng))

    model = CCCPR(
        vocab_size=len(global_vocab),
        feature_dim=features.FEATURE_DIM,
        repr_dim=args.repr_dim,
        hidden=args.hidden,
        num_packs=cfg.num_packs,
        pack_size=cfg.pack_size,
        alpha_floor=args.alpha_floor,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        state = ckpt.load_checkpoint(args.resume, args.ckpt_dir, device=str(device))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        logger.info("resumed from step %d", start_step)

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        job_type="stage2-distill",
        config={
            **vars(args),
            "params": sum(p.numel() for p in model.parameters()),
            "feature_dim": features.FEATURE_DIM,
            "vocab_size": len(global_vocab),
        },
    )

    logger.info("training params: %d", sum(p.numel() for p in model.parameters()))
    for step in range(start_step, args.steps):
        # Anneal alpha 1.0 -> floor over alpha_decay_steps.
        frac = min(1.0, step / max(1, args.alpha_decay_steps))
        model.set_alpha(1.0 - (1.0 - args.alpha_floor) * frac)

        ci = step % args.num_cubes
        data = train_data[ci]
        idx = rng.integers(0, len(data["picked"]), size=args.batch_size)
        batch, picked = make_batch(data, idx, device, args.mask_cube, rng)

        model.train()
        scores = model(batch, cube_ts[ci])
        loss = triplet_loss(scores, batch.pack, picked, margin=args.margin)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            logger.info("step %d/%d  loss=%.4f  alpha=%.3f", step, args.steps, loss.item(), float(model.alpha))
            wandb.log({"step": step, "loss": loss.item(), "alpha": float(model.alpha)}, step=step)

        if step > start_step and step % args.eval_interval == 0:
            metrics = _eval_all(model, cube_ts, eval_data, device)
            logger.info(
                "step %d  eval top1=%.3f top3=%.3f mrr=%.3f (random top1=%.3f)",
                step, metrics["top1"], metrics["top3"], metrics["mrr"], metrics["random_top1"],
            )
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

        if step > start_step and step % args.ckpt_interval == 0:
            _save(model, optimizer, step, args)

    # Final eval + checkpoint.
    metrics = _eval_all(model, cube_ts, eval_data, device)
    logger.info("FINAL eval top1=%.3f top3=%.3f mrr=%.3f (random=%.3f)",
                metrics["top1"], metrics["top3"], metrics["mrr"], metrics["random_top1"])
    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=args.steps)
    for k, v in metrics.items():
        wandb.summary[f"final_{k}"] = v
    _save(model, optimizer, args.steps, args)
    run.finish()

    print(
        f"STAGE2 DONE  device={device.type}  steps={args.steps}  "
        f"top1={metrics['top1']:.3f}  top3={metrics['top3']:.3f}  mrr={metrics['mrr']:.3f}  "
        f"(random top1={metrics['random_top1']:.3f})"
    )


def _eval_all(model, cube_ts, eval_data, device) -> dict[str, float]:
    per_cube = [evaluate(model, cube_ts[i], eval_data[i], device) for i in range(len(eval_data))]
    return {k: float(np.mean([m[k] for m in per_cube])) for k in per_cube[0]}


def _save(model, optimizer, step: int, args: argparse.Namespace) -> None:
    ckpt.save_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        },
        args.ckpt_dir,
        step,
        r2_prefix=args.r2_prefix,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/cards.parquet"))
    p.add_argument("--cube-size", type=int, default=360)
    p.add_argument("--num-cubes", type=int, default=4, help="distinct random cubes to train across")
    p.add_argument("--drafts-per-cube", type=int, default=64, help="teacher drafts generated per cube")
    p.add_argument("--eval-drafts", type=int, default=16, help="held-out teacher drafts per cube for eval")
    p.add_argument("--teacher", default="raredraft", choices=["raredraft", "random"])

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.2, help="triplet loss margin")
    p.add_argument("--mask-cube", action="store_true", help="§2.5 random cube-list masking")

    p.add_argument("--repr-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--alpha-floor", type=float, default=0.3)
    p.add_argument("--alpha-decay-steps", type=int, default=2000)

    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/stage2"))
    p.add_argument("--r2-prefix", default=None, help="R2 key prefix to mirror checkpoints, e.g. checkpoints/stage2/run1")
    p.add_argument("--resume", default=None, help="checkpoint to resume: local path, R2 key, or 'latest'")

    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="cube-draft")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.data.exists():
        raise FileNotFoundError(
            f"card data not found at {args.data}. Run scripts/download_scryfall.py or "
            f"scripts/r2_sync.py pull scryfall/oracle_cards.parquet {args.data}."
        )
    train(args)


if __name__ == "__main__":
    main()
