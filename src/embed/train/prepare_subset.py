import argparse
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(
        description="Download AudioSetCaps and prepare a subset."
    )
    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=5000,
        help="Number of samples to include in the subset (default: 5000)",
    )
    args = parser.parse_args()

    # Fallback to local './scratch' if not on RCAC
    scratch_base = os.environ.get("RCAC_SCRATCH", "./scratch")
    out_dir = Path(scratch_base) / "engine" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {out_dir}")

    print("Downloading AudioSetCaps_caption.csv from HuggingFace...")
    # hf_hub_download will download the file. Using local_dir to control destination.
    downloaded_path = hf_hub_download(
        repo_id="baijs/AudioSetCaps",
        filename="Dataset/AudioSetCaps_caption.csv",
        repo_type="dataset",
        local_dir=out_dir,
    )

    # Move the file from data/Dataset/AudioSetCaps_caption.csv to data/AudioSetCaps_caption.csv
    src = Path(downloaded_path)
    dest = out_dir / "AudioSetCaps_caption.csv"

    if src.exists() and src != dest:
        src.replace(dest)
        # Clean up the empty Dataset directory if it exists
        try:
            src.parent.rmdir()
        except OSError:
            pass

    if not dest.exists():
        raise FileNotFoundError(f"Could not find downloaded CSV at {dest}")

    print(f"Reading dataset from {dest}...")
    df = pd.read_csv(dest)
    print(f"Original dataset size: {len(df)}")

    n_samples = min(args.n_samples, len(df))
    print(f"Sampling {n_samples} random rows with seed 42...")
    subset = df.sample(n=n_samples, random_state=42)

    subset_path = out_dir / "AudioSetCaps_caption_subset.csv"
    subset.to_csv(subset_path, index=False)
    print(f"Created subset with {len(subset)} rows at {subset_path}")


if __name__ == "__main__":
    main()
