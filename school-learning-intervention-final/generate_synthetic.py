"""
generate_synthetic.py -- entry point.

Run from the project root:
    python -m src.data_generation.generate_synthetic
    python -m src.data_generation.generate_synthetic --seed 7 --out-dir data

It only wires the two layers together:
    generators.generate_all(seed)  ->  writer.write_tables(...)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import scenarios as sc
from .generators import generate_all
from .writer import write_tables

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic school dataset.")
    parser.add_argument("--seed", type=int, default=sc.SEED, help="random seed (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data",
                        help="folder that will contain raw/ and ground_truth/")
    args = parser.parse_args()

    tables = generate_all(seed=args.seed)
    written = write_tables(tables, args.out_dir / "raw", args.out_dir / "ground_truth")

    print(f"Seed {args.seed}. Wrote {len(written)} files under {args.out_dir}:")
    for path in written:
        name = path.stem
        print(f"  {path.relative_to(args.out_dir)!s:<40} {len(tables[name]):>5} rows")


if __name__ == "__main__":
    main()
