import argparse
import subprocess
from pathlib import Path

import pandas as pd


# TODO: find a way to associate your account with yt-dlp
# a lot of the request start getting denied if you don't
# which would lead to no audio clip gettind downloaded.
def fetch_audio(youtube_id, start_sec, duration, out_file, skip_existing=True):
    """
    Downloads and trims audio from a YouTube video.
    """
    if skip_existing and out_file.exists():
        print(f"Skipping {youtube_id}, file already exists at {out_file}")
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    print(
        f"Fetching audio for {youtube_id} ({start_sec}s to {start_sec + duration}s)..."
    )

    # Using yt-dlp to download and ffmpeg via postprocessor to trim/resample
    # -ar 16000: 16kHz
    # -ac 1: mono
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar 16000 -ac 1 -ss {start_sec} -t {duration}",
        "-o",
        str(out_file.with_suffix(".%(ext)s")),
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully saved to {out_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"Failed to download {youtube_id}. The video might be unavailable or private."
        )
        return False


def batch_fetch(csv_path, audio_dir, limit=None, skip_existing=True):
    """
    Batch downloads audio samples from a CSV file.
    """
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    audio_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    for _, row in df.iterrows():
        youtube_id = row.get("id")
        if youtube_id and youtube_id.startswith("Y"):
            youtube_id = youtube_id[1:]

        if not youtube_id:
            print(f"Warning: Skipping row with missing YouTube ID.")
            continue

        start_sec = int(row.get("start_time", 0))
        end_sec = int(row.get("end_time", start_sec + 10))
        duration = max(1, end_sec - start_sec)

        out_file = audio_dir / f"{youtube_id}.wav"
        fetch_audio(
            youtube_id, start_sec, duration, out_file, skip_existing=skip_existing
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch audio samples from YouTube.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/AudioSetCaps_caption_subset.csv",
        help="Path to the subset CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/audio",
        help="Directory to save audio files.",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of samples to download."
    )
    parser.add_argument(
        "--id", type=str, help="Specific YouTube ID to download (ignores CSV)."
    )
    parser.add_argument(
        "--no_skip",
        action="store_false",
        dest="skip_existing",
        help="Do not skip existing files.",
    )
    parser.set_defaults(skip_existing=True)

    args = parser.parse_args()

    csv_path = Path(args.csv)
    audio_dir = Path(args.out_dir)

    if args.id:
        out_file = audio_dir / f"{args.id}.wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        # For a single ID, we'll use default 0s start and 10s duration if not in CSV context
        # In a real batching scenario, we'd want more metadata.
        fetch_audio(args.id, 0, 10, out_file, skip_existing=args.skip_existing)
    else:
        batch_fetch(
            csv_path, audio_dir, limit=args.limit, skip_existing=args.skip_existing
        )
