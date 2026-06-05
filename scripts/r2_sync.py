"""R2 sync CLI.

Examples:
    # From your laptop after pulling Scryfall data locally:
    python scripts/r2_sync.py push data/cards.parquet scryfall/oracle_cards.parquet

    # On a fresh RunPod machine:
    python scripts/r2_sync.py pull scryfall/oracle_cards.parquet data/cards.parquet

    # List everything in the bucket:
    python scripts/r2_sync.py list

    # Recursively push a directory:
    python scripts/r2_sync.py push-dir data/cubecobra_audit cubecobra/audit

    # Pull a whole prefix:
    python scripts/r2_sync.py pull-prefix scryfall/ data/scryfall/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()  # pull .env into os.environ for local dev
except ImportError:
    pass  # python-dotenv not installed; assume env is already set

from cube_draft.utils import r2  # noqa: E402


def cmd_push(args: argparse.Namespace) -> None:
    r2.push(Path(args.local), args.remote)


def cmd_pull(args: argparse.Namespace) -> None:
    r2.pull(args.remote, Path(args.local))


def cmd_list(args: argparse.Namespace) -> None:
    objs = r2.list_prefix(args.prefix)
    for o in objs:
        print(f"{o['Size']:>12d}  {o['Key']}")
    print(f"{len(objs)} objects, {sum(o['Size'] for o in objs)} bytes total")


def cmd_push_dir(args: argparse.Namespace) -> None:
    n = r2.push_dir(Path(args.local_dir), args.remote_prefix)
    print(f"pushed {n} files to r2://{r2.bucket_name()}/{args.remote_prefix}/")


def cmd_pull_prefix(args: argparse.Namespace) -> None:
    n = r2.pull_prefix(args.remote_prefix, Path(args.local_dir))
    print(f"pulled {n} files from r2://{r2.bucket_name()}/{args.remote_prefix}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    push_p = sub.add_parser("push", help="upload one file")
    push_p.add_argument("local")
    push_p.add_argument("remote")
    push_p.set_defaults(fn=cmd_push)

    pull_p = sub.add_parser("pull", help="download one file")
    pull_p.add_argument("remote")
    pull_p.add_argument("local")
    pull_p.set_defaults(fn=cmd_pull)

    list_p = sub.add_parser("list", help="list objects under a prefix")
    list_p.add_argument("prefix", nargs="?", default="")
    list_p.set_defaults(fn=cmd_list)

    pd = sub.add_parser("push-dir", help="recursively upload a directory")
    pd.add_argument("local_dir")
    pd.add_argument("remote_prefix")
    pd.set_defaults(fn=cmd_push_dir)

    pp = sub.add_parser("pull-prefix", help="download every object under a prefix")
    pp.add_argument("remote_prefix")
    pp.add_argument("local_dir")
    pp.set_defaults(fn=cmd_pull_prefix)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
