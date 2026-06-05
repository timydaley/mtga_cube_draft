"""Generate a CubeCobraBot distillation dataset (plan §4.5 Stage 2 teacher).

Runs all-CubeCobraBot drafts on N random cubes, records every pick decision, and
writes a dataset (mirrored to R2) that scripts/train_stage2.py consumes via
--dataset. This is the offline, run-once step that keeps the AGPL Node teacher
OUT of the training inner loop (plan §4.7.4).

Slow by design: each pick is a round-trip to the Node/WASM bot, so a draft is
~360 calls. Keep --drafts-per-cube modest; scale cubes/drafts later.

Usage:
    # one-time setup: node + the package (cloud_bootstrap.sh does this on the pod)
    npm install
    python scripts/gen_distill_dataset.py --num-cubes 4 --drafts-per-cube 32 \
        --out data/datasets/cubecobra_v1 --r2-prefix datasets/cubecobra_v1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:
    pass

from cube_draft.bots.cubecobra import CubeCobraBot, NodeBridgeTransport  # noqa: E402
from cube_draft.cards.vocab import CardVocab, CubeVocab  # noqa: E402
from cube_draft.data import (
    cube_list,  # noqa: E402
    rollout,  # noqa: E402
)
from cube_draft.data import dataset as ds  # noqa: E402
from cube_draft.env.cube_draft import DraftConfig  # noqa: E402

logger = logging.getLogger("gen_distill_dataset")

_BASIC_NAMES = ("Plains", "Island", "Swamp", "Mountain", "Forest")


def find_basics(vocab: CardVocab) -> list[str]:
    """Oracle ids of the five basic lands, for the bot's mana/land evaluation."""
    out: list[str] = []
    for name in _BASIC_NAMES:
        try:
            out.append(vocab.card(vocab.index_by_name(name)).oracle_id)
        except KeyError:
            logger.warning("basic land %r not in vocab; bot mana eval may degrade", name)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/cards.parquet"))
    p.add_argument("--out", type=Path, required=True, help="dataset output directory")
    p.add_argument("--r2-prefix", default=None, help="R2 prefix to mirror the dataset to")
    p.add_argument(
        "--cube-list",
        default=None,
        help="oracle_id cube file (path or R2 key) to generate on a fixed published "
        "cube (e.g. the Arena Cube) instead of random cubes; sets num-cubes=1",
    )
    p.add_argument("--num-cubes", type=int, default=4)
    p.add_argument("--cube-size", type=int, default=360)
    p.add_argument("--drafts-per-cube", type=int, default=32)
    p.add_argument("--eval-drafts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--node", default="node", help="node executable")
    p.add_argument("--no-recognized-check", action="store_true", help="skip testRecognized")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.data.exists():
        raise FileNotFoundError(f"card data not found at {args.data}")

    rng = np.random.default_rng(args.seed)
    vocab = CardVocab.from_parquet(args.data)
    basics = find_basics(vocab)
    cfg = DraftConfig()

    if args.cube_list:
        fixed = cube_list.load_oracle_ids(args.cube_list)
        cube_specs = [fixed]
        logger.info("cube-list %s: %d oracle_ids", args.cube_list, len(fixed))
    else:
        cube_specs = [
            [vocab.card(int(i)).oracle_id for i in rng.choice(len(vocab), size=min(args.cube_size, len(vocab)), replace=False)]
            for _ in range(args.num_cubes)
        ]

    transport = NodeBridgeTransport(node=args.node)
    try:
        cubes_data: list[ds.CubeData] = []
        for c, oracle_ids in enumerate(cube_specs):
            cube = CubeVocab(vocab, oracle_ids)

            bot = CubeCobraBot(
                cube,
                transport=transport,
                basics=basics,
                check_recognized=not args.no_recognized_check,
            )
            drafters = [bot] * cfg.num_seats  # stateless across seats; one bot is fine
            logger.info("cube %d/%d (%d cards): %d train + %d eval drafts",
                        c + 1, len(cube_specs), cube.size, args.drafts_per_cube, args.eval_drafts)
            train = rollout.collect_decisions(cube, drafters, args.drafts_per_cube, cfg, rng)
            eval_ = rollout.collect_decisions(cube, drafters, args.eval_drafts, cfg, rng)
            cube_oracle_ids = [cube.oracle_id(i) for i in range(cube.size)]
            cubes_data.append(ds.CubeData(oracle_ids=cube_oracle_ids, train=train, eval=eval_))
    finally:
        transport.close()

    ds.save_dataset(args.out, "cubecobra", cubes_data, config=vars(args) | {"data": str(args.data)}, r2_prefix=args.r2_prefix)
    total = sum(len(cd.train["picked"]) for cd in cubes_data)
    print(f"GEN DONE  cubes={len(cubes_data)}  train_decisions={total}  out={args.out}")


if __name__ == "__main__":
    main()
