"""
Build the small, committed sample from the real German Credit file.

Why this script exists
----------------------
The repo must run on a fresh clone with no downloads — so a tiny sample is
committed. But a hand-typed sample is fiction, and fiction hides real data
problems. This script cuts the sample out of the REAL file, stratified on the
target and with a fixed seed, so it is representative, reproducible, and
regenerable by anyone.

Usage
-----
    python scripts/make_sample.py                  # uses config.yaml settings
    python scripts/make_sample.py --rows 50        # a bigger sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed
from src import load


def main():
    parser = argparse.ArgumentParser(description="Build the bundled sample.")
    parser.add_argument("--rows", type=int, default=None, help="Sample size")
    args = parser.parse_args()

    config = load_config()
    seed = config.get("seed", 42)
    set_seed(seed)

    data_cfg = config["data"]
    raw_path = REPO_ROOT / data_cfg["raw_dir"] / data_cfg["raw_file"]
    out_path = REPO_ROOT / data_cfg["sample_path"]
    n_rows = args.rows or data_cfg.get("sample_rows", 24)

    if not raw_path.exists():
        sys.exit(
            f"Raw file not found: {raw_path}\n"
            f"Download it first: {data_cfg['download_url']}"
        )

    df = load.load_german_raw(raw_path, config)

    # Stratified: keep the real good/bad mix, so the sample's target balance is
    # not an accident of which rows happened to come first. We collect row
    # labels group by group rather than using groupby().apply(), which in
    # pandas 2.x drops the grouping column from the result.
    target = config["schema"]["target_column"]
    fraction = n_rows / len(df)

    keep = []
    for value, group in df.groupby(target):
        take = max(1, round(len(group) * fraction))   # at least one of each class
        keep.extend(group.sample(take, random_state=seed).index)

    sample = df.loc[sorted(keep)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)

    counts = sample[target].value_counts().sort_index().to_dict()
    print(f"Wrote {len(sample)} rows to {out_path.relative_to(REPO_ROOT)}")
    print(f"target balance (1=good, 2=bad): {counts}")
    print(f"seed={seed} — rerunning this command reproduces the same sample.")


if __name__ == "__main__":
    main()
