"""Fetch 17lands cube card ratings -> oracle-keyed teacher signal (plan §4.2).

Pulls per-card metrics for the Arena Powered Cube from 17lands' card_ratings API
(real human draft data: win-rate contribution, GIH WR, ATA pick-order), joins
them to our Scryfall vocab by name, and writes an oracle-keyed JSON that
train_stage2.py uses as the --teacher 17lands signal. Mirrors to R2.

Usage:
    python scripts/fetch_17lands_cube.py --out data/17lands/cube_powered.json \
        --r2-key 17lands/cube_powered.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:
    pass

from cube_draft.cards.vocab import CardVocab  # noqa: E402
from cube_draft.data import seventeenlands as sl  # noqa: E402

logger = logging.getLogger("fetch_17lands_cube")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/cards.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/17lands/cube_powered.json"))
    p.add_argument("--r2-key", default=None, help="optional R2 key to upload to")
    p.add_argument("--expansion", default=sl.DEFAULT_EXPANSION)
    p.add_argument("--format", default=sl.DEFAULT_FORMAT)
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2026-06-06")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.data.exists():
        raise FileNotFoundError(f"card data not found at {args.data}")

    rows = sl.fetch_card_ratings(args.expansion, args.format, args.start, args.end)
    logger.info("fetched %d cards from 17lands (%s / %s)", len(rows), args.expansion, args.format)
    vocab = CardVocab.from_parquet(args.data)
    by_oracle = sl.ratings_by_oracle(rows, vocab)
    logger.info("joined %d/%d cards to the vocab by name", len(by_oracle), len(rows))

    sl.save_ratings(
        by_oracle,
        args.out,
        meta={"expansion": args.expansion, "format": args.format, "start": args.start, "end": args.end},
    )
    logger.info("wrote %s", args.out)

    if args.r2_key:
        from cube_draft.utils import r2

        r2.push(args.out, args.r2_key)
        logger.info("uploaded to r2://%s/%s", r2.bucket_name(), args.r2_key)

    print(f"FETCH DONE  cards={len(rows)}  joined={len(by_oracle)}  out={args.out}")


if __name__ == "__main__":
    main()
