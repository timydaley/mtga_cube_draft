"""Resolve a published cube card-name list to an oracle_id cube file (plan §2.5).

Reads a newline-delimited card-name file (e.g. cubes/arena_cube_5_0.provisional.txt,
scraped from magic.wizards.com), matches names against the Scryfall vocab, and
writes a JSON list of oracle_ids that gen_distill_dataset.py / train_stage2.py
consume via --cube-list. Unresolved names are written to <out>.unmatched.txt so
the source list can be fixed — this is the QA gate for a scraped list.

Usage:
    python scripts/ingest_cube_list.py --names cubes/arena_cube_5_0.provisional.txt \
        --out cubes/arena_cube_5_0.json [--r2-key cubes/arena_cube_5_0.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:
    pass

from cube_draft.cards.vocab import CardVocab  # noqa: E402
from cube_draft.data import cube_list  # noqa: E402

logger = logging.getLogger("ingest_cube_list")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", type=Path, required=True, help="newline-delimited card-name file")
    p.add_argument("--data", type=Path, default=Path("data/cards.parquet"))
    p.add_argument("--out", type=Path, required=True, help="output oracle_id JSON")
    p.add_argument("--r2-key", default=None, help="optional R2 key to upload the cube to")
    p.add_argument("--expect", type=int, default=540, help="expected card count (warns if off)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.data.exists():
        raise FileNotFoundError(f"card data not found at {args.data}")

    names = cube_list.parse_name_file(args.names.read_text())
    vocab = CardVocab.from_parquet(args.data)
    res = cube_list.resolve_names(vocab, names)

    logger.info(
        "names=%d  matched=%d  unmatched=%d  duplicates=%d  -> %d oracle_ids",
        len(names), len(res.matched), len(res.unmatched), len(res.duplicates), len(res.oracle_ids),
    )
    if res.unmatched:
        unmatched_path = args.out.with_suffix(".unmatched.txt")
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        unmatched_path.write_text("\n".join(res.unmatched) + "\n")
        logger.warning("%d names did not resolve — see %s", len(res.unmatched), unmatched_path)
    if res.duplicates:
        logger.warning("%d duplicate names skipped: %s", len(res.duplicates), ", ".join(res.duplicates[:10]))
    if len(res.oracle_ids) != args.expect:
        logger.warning("resolved %d cards, expected %d", len(res.oracle_ids), args.expect)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res.oracle_ids, indent=0))
    logger.info("wrote %d oracle_ids to %s", len(res.oracle_ids), args.out)

    if args.r2_key:
        from cube_draft.utils import r2

        r2.push(args.out, args.r2_key)
        logger.info("uploaded to r2://%s/%s", r2.bucket_name(), args.r2_key)

    print(f"INGEST DONE  resolved={len(res.oracle_ids)}/{len(names)}  unmatched={len(res.unmatched)}  out={args.out}")


if __name__ == "__main__":
    main()
