from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading AudioSetCaps_caption.csv from HuggingFace...")
snapshot_download(
    repo_id="baijs/AudioSetCaps",
    repo_type="dataset",
    local_dir=str(out_dir),
    allow_patterns=["Dataset/AudioSetCaps_caption.csv"],
)

src = out_dir / "Dataset" / "AudioSetCaps_caption.csv"
if not src.exists():
    raise FileNotFoundError(f"Could not find downloaded CSV at {src}")

df = pd.read_csv(src)
print(f"Original dataset size: {len(df)}")

subset = df.head(min(100, len(df)))
subset_path = out_dir / "AudioSetCaps_caption_subset.csv"
subset.to_csv(subset_path, index=False)
print(f"Created subset with {len(subset)} rows at {subset_path}")
