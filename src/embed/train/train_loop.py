import math
import sys
from pathlib import Path

# Add the project root to the PYTHONPATH so 'src' can be found
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.embed.audio import AudioEmbedder
from src.embed.qwen import QwenEmbedder

CSV_PATH = "data/AudioSetCaps_caption_subset.csv"
AUDIO_DIR = "data/audio"
CHECKPOINT_DIR = "data/checkpoints"
BATCH_SIZE = 4  # Start small; on H100, increase dramatically to 128/256
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.2


class AudioTextDataset(Dataset):
    def __init__(self, csv_file: str, audio_dir: str):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = Path(audio_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        yt_id = row.get("id", str(row.name))

        # AudioSet dataset specific formatting handling
        if isinstance(yt_id, str) and yt_id.startswith("Y"):
            yt_id = yt_id[1:]

        audio_path = self.audio_dir / f"{yt_id}.wav"

        # TODO: fix this, it shouldn't ever reach this scenario
        # where we have a missing audio file. Happens because some of the youtube
        # links are no longer valid. Maybe create a new file that filters out the
        # dataframe even further to only store valid pairs of (audio_file, caption)
        # Then that would reduce the complexity of this data loader.
        #
        # Load audio (fallback to silence if missing to keep batching intact)
        if audio_path.exists():
            # Use fixed sr and mono for speed
            audio_array, _ = librosa.load(audio_path, sr=48000, mono=True)
        else:
            audio_array = np.zeros(48000, dtype=np.float32)

        return audio_array, str(row["caption"])


def collate_fn(batch):
    audios, texts = zip(*batch)
    return list(audios), list(texts)

# TODO: add a way to store / display logs of the training
# loop as it is running. Will help if we need to stop early
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading Base Models...")
    qwen = QwenEmbedder(dtype=torch.bfloat16)  # Generates text targets
    aligner = AudioEmbedder(device=device, dtype=torch.bfloat16)

    # Extract and Isolate the Trainable Projection Head
    proj_head = aligner.projection_head()
    proj_head.train()

    # Learnable contrastive temperature scalar
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))

    optimizer = torch.optim.AdamW(
        [{"params": proj_head.parameters()}, {"params": [logit_scale]}],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Data Supply
    num_workers = 4 if torch.cuda.is_available() else 0
    dataset = AudioTextDataset(CSV_PATH, AUDIO_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    print(f"Starting Training! Batches per epoch: {len(loader)}")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        for audios, texts in loader:
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            with torch.no_grad():
                # Text Targets
                text_embeds = qwen.embed_batch(texts).to(device, dtype=torch.bfloat16)

                # Audio Encoding via CLAP
                inputs = aligner._processor(
                    audio=audios,
                    sampling_rate=aligner.CLAP_SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                input_features = inputs["input_features"].to(
                    device, dtype=torch.bfloat16
                )

                audio_outputs = aligner._audio_model(input_features=input_features)
                clap_embed = aligner._audio_projection(audio_outputs.pooler_output)
                clap_embed = F.normalize(clap_embed, p=2, dim=-1)

            # Cast using AMP
            device_type_str = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type_str, dtype=torch.bfloat16):
                audio_embeds = proj_head(clap_embed.float())

                # Contrastive (Symmetric InfoNCE) Loss
                scale = torch.clamp(logit_scale.exp(), max=100.0)
                logits_per_audio = scale * audio_embeds @ text_embeds.t()
                logits_per_text = logits_per_audio.t()

                labels = torch.arange(len(audio_embeds), device=device)
                loss_a = F.cross_entropy(logits_per_audio, labels)
                loss_t = F.cross_entropy(logits_per_text, labels)
                loss = (loss_a + loss_t) / 2.0

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f} - lr: {current_lr:.2e} - temp: {scale.item():.2f}"
        )

        scheduler.step()

        ckpt_path = Path(CHECKPOINT_DIR) / f"proj_head_ep{epoch + 1}.pt"
        torch.save(proj_head.state_dict(), ckpt_path)
        print(f"-> Saved Checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
