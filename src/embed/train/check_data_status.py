import os
from pathlib import Path

import pandas as pd


def check_status():
    scratch_base = os.environ.get("RCAC_SCRATCH", "./scratch")
    csv_path = (
        Path(scratch_base) / "engine" / "data" / "AudioSetCaps_caption_subset.csv"
    )
    audio_dir = Path(scratch_base) / "engine" / "audio"

    print("--- Data Status Report ---")
    print(f"Scratch Base: {scratch_base}")

    if not csv_path.exists():
        print(f"Error: Subset CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_target = len(df)
    print(f"Total entries in CSV: {total_target}")

    if not audio_dir.exists():
        print(f"Audio directory does not exist yet: {audio_dir}")
        audio_count = 0
    else:
        audio_files = list(audio_dir.glob("*.wav"))
        audio_count = len(audio_files)

    print(f"Total .wav files found: {audio_count}")

    percent = (audio_count / total_target) * 100 if total_target > 0 else 0
    print(f"Completion: {audio_count}/{total_target} ({percent:.1f}%)")

    if audio_count < total_target:
        print(f"Remaining: {total_target - audio_count} files left to download.")
    else:
        print("Dataset is fully complete!")


if __name__ == "__main__":
    check_status()
