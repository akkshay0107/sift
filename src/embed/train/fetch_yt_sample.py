import argparse
import os
import random
import subprocess
import time
from pathlib import Path

import pandas as pd


def fetch_audio(
    youtube_id,
    start_sec,
    duration,
    out_file,
    skip_existing=True,
    browser=None,
    cookies=None,
):
    """
    Downloads and trims audio from a YouTube video.
    """
    if skip_existing and out_file.exists():
        print(f"Skipping {youtube_id}, file already exists at {out_file}")
        return "skip"

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    print(
        f"Fetching audio for {youtube_id} ({start_sec}s to {start_sec + duration}s)..."
    )

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
    ]

    if cookies:
        cmd.extend(["--cookies", cookies])
    elif browser:
        cmd.extend(["--cookies-from-browser", browser])

    cmd.extend(
        [
            "--remote-components",
            "ejs:github",
            "--postprocessor-args",
            f"ffmpeg:-ar 48000 -ac 1 -ss {start_sec} -t {duration}",
            "-o",
            str(out_file.with_suffix(".%(ext)s")),
            url,
        ]
    )

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"Successfully saved to {out_file}")
        return "success"
    except subprocess.CalledProcessError:
        print(
            f"Failed to download {youtube_id}. The video might be unavailable or private."
        )
        return "fail"


def batch_fetch(
    csv_path,
    audio_dir,
    offset=0,
    limit=None,
    skip_existing=True,
    browser=None,
    cookies=None,
):
    """
    Batch downloads audio samples from a CSV file.
    """
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate Schema before fetching
    if "id" not in df.columns:
        print("Error: CSV must contain an 'id' column.")
        return
    if "caption" not in df.columns:
        print("Error: CSV must contain a 'caption' column.")
        return

    if offset > 0:
        df = df.iloc[offset:]
        print(f"Applying offset: Skipping first {offset} rows.")

    if limit:
        df = df.head(limit)
        print(f"Limiting fetch to {limit} rows.")

    total = len(df)
    start_time = time.time()
    stats = {"success": 0, "fail": 0, "skip": 0}

    # Randomize the first sleep trigger between 10 and 20
    next_sleep_at = random.randint(10, 20)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        # Force the ID into a string to avoid weird edge cases
        youtube_id = str(row["id"])

        # Strip potential audio-set prefix
        if youtube_id.startswith("Y"):
            youtube_id = youtube_id[1:]

        if not youtube_id.strip():
            print("Warning: Skipping row with empty YouTube ID.")
            continue

        start_sec = int(row.get("start_time", 0))
        end_sec = int(row.get("end_time", start_sec + 10))
        duration = max(1, end_sec - start_sec)

        out_file = audio_dir / f"{youtube_id}.wav"
        status = fetch_audio(
            youtube_id,
            start_sec,
            duration,
            out_file,
            skip_existing=skip_existing,
            browser=browser,
            cookies=cookies,
        )

        if status in stats:
            stats[status] += 1

        # Log progress every 10 items or on the last item
        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = elapsed / i
            remaining_secs = rate * (total - i)

            m, s = divmod(int(remaining_secs), 60)
            h, m = divmod(m, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

            print(
                f"\n--- Progress: {i}/{total} ({(i / total) * 100:.1f}%) "
                f"| Elapsed: {int(elapsed)}s | ETA: {time_str} ---"
            )
            print(
                f"--- Stats: {stats['success']} Success | {stats['skip']} Skipped | {stats['fail']} Failed ---\n"
            )

        # Sleep logic: Randomized break every ~10-20 downloads
        if i == next_sleep_at and i < total:
            sleep_dur = random.uniform(5, 12)
            print(f"--- Anti-Throttling: Taking a {sleep_dur:.1f}s breather... ---\n")
            time.sleep(sleep_dur)
            next_sleep_at += random.randint(10, 20)


if __name__ == "__main__":
    scratch_base = os.environ.get("RCAC_SCRATCH", "./scratch")
    default_csv = (
        Path(scratch_base) / "engine" / "data" / "AudioSetCaps_caption_subset.csv"
    )
    default_out_dir = Path(scratch_base) / "engine" / "audio"

    parser = argparse.ArgumentParser(description="Fetch audio samples from YouTube.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help="Path to the subset CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(default_out_dir),
        help="Directory to save audio files.",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset row to start fetching from."
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of samples to download."
    )
    parser.add_argument(
        "--id", type=str, help="Specific YouTube ID to download (ignores CSV)."
    )
    parser.add_argument(
        "--browser",
        type=str,
        help="Browser to extract cookies from (e.g., chrome, safari, brave, firefox, edge) to avoid bot detection.",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        help="Path to a Netscape-format cookies.txt file exported from your browser.",
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

    print(f"Output directory set to: {audio_dir}")
    if args.id:
        out_file = audio_dir / f"{args.id}.wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        fetch_audio(
            args.id,
            0,
            10,
            out_file,
            skip_existing=args.skip_existing,
            browser=args.browser,
            cookies=args.cookies,
        )
    else:
        batch_fetch(
            csv_path,
            audio_dir,
            offset=args.offset,
            limit=args.limit,
            skip_existing=args.skip_existing,
            browser=args.browser,
            cookies=args.cookies,
        )
