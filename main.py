#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from registry import BLOCKS
from utils import parse_extras

# Register blocks
import pipeline  # registers "pipeline"
import blocks    # registers glctx/glprogram/glcube/glloop
import builder_blocks

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="block-cli",
        description="Modular block-processing CLI. Choose a block and pass a prompt."
    )
    available = ", ".join(BLOCKS.names()) or "(none)"
    p.add_argument("block", help=f"Block to run. Available: {available}")
    p.add_argument("prompt", nargs="?", default=None, help="Input payload (or stdin if omitted)")
    p.add_argument("--extra", action="append", default=[], help="key=val (supports name.key=val, all.key=val)")
    p.add_argument("--json", action="store_true", help="Print JSON with metadata")
    return p


def read_prompt(arg: Optional[str]) -> str:
    if arg is not None:
        return arg
    if sys.stdin.isatty():
        print("Enter payload then Ctrl+D:", file=sys.stderr)
    return sys.stdin.read()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        extras = parse_extras(args.extra)
    except Exception as e:
        parser.error(f"Failed to parse --extra: {e}")
        return 2

    payload = read_prompt(args.prompt)
    if not payload and args.prompt is None:
        parser.error("No prompt provided via arg or stdin.")
        return 1

    try:
        block = BLOCKS.create(args.block)

        # Merge “all” with block-specific, block wins
        params: Dict[str, Any] = {}
        params.update(extras.get("all", {}))
        params.update(extras.get(args.block.lower(), {}))

        result, meta = block.execute(payload, params=params)

        if args.json:
            print(json.dumps({"block": args.block, "metadata": meta, "result": result}, indent=2, ensure_ascii=False))
        else:
            print(result, end="")
        return 0

    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except NotImplementedError:
        print(f"Error: Block '{args.block}' is missing an 'execute' implementation.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error in block '{args.block}': {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
