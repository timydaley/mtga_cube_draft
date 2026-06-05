"""Download Scryfall oracle_cards and persist to Parquet.

Usage:
    python scripts/download_scryfall.py [--out data/cards.parquet]

Run weekly. Cloud usage: invoke from the GPU machine's setup script to
populate the local cache before training starts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cube_draft.cards.scryfall import (
    cards_to_parquet,
    download_oracle_cards,
    parse_oracle_cards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/cards.parquet"),
        help="Output Parquet path",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/scryfall_oracle_cards.json"),
        help="Path to cache raw Scryfall JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_oracle_cards(args.raw)
    cards = parse_oracle_cards(args.raw)
    cards_to_parquet(cards, args.out)
    print(f"wrote {len(cards)} cards to {args.out}")


if __name__ == "__main__":
    main()
