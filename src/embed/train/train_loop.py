import math
import os
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
from tqdm import tqdm

from src.embed.audio import AudioEmbedder
from src.embed.qwen import QwenEmbedder

scratch_base = os.environ.get("RCAC_SCRATCH", "./scratch")
CSV_PATH = os.environ.get("CSV_PATH", str(Path(scratch_base) / "engine" / "data" / "AudioSetCaps_caption_subset.csv"))
AUDIO_DIR = os.environ.get("AUDIO_DIR", str(Path(scratch_base) / "engine" / "audio"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", str(Path(scratch_base) / "engine" / "checkpoints"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
EPOCHS = int(os.environ.get("EPOCHS", 50))
LR = float(os.environ.get("LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.2))


class AudioTextDataset(Dataset):
    def __init__(self, csv_file: str, audio_dir: str):
        print(f"Loading CSV from: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.audio_dir = Path(audio_dir)
        
        print("Verifying audio files exist to prevent missing samples...")
        valid_rows = []
        for idx, row in self.df.iterrows():
            yt_id = str(row.get("id", str(row.name)))
            if yt_id.startswith("Y"):
                yt_id = yt_id[1:]
            
            audio_path = self.audio_dir / f"{yt_id}.wav"
            if audio_path.exists():
                valid_rows.append(idx)
        
        original_len = len(self.df)
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(f"Retained {len(self.df)} valid pairs from {original_len} total rows.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        yt_id = str(row.get("id"))
        if yt_id.startswith("Y"):
            yt_id = yt_id[1:]

        audio_path = self.audio_dir / f"{yt_id}.wav"
        
        try:
            # 48kHz mono loading
            audio_array, _ = librosa.load(audio_path, sr=48000, mono=True)
        except Exception as e:
            print(f"Warning: Failed to load {audio_path}: {e}")
            audio_array = np.zeros(48000, dtype=np.float32)

        return audio_array, str(row["caption"])


def collate_fn(batch):
    audios, texts = zip(*batch)
    return list(audios), list(texts)


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using compute device: {device}")

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be managed in: {CHECKPOINT_DIR}")

    print("Initializing Base Models...")
    
    # Safe float fallback for CPU/MPS locally, bfloat16 for CUDA
    dtype = torch.float32 if device.type in ["cpu", "mps"] else torch.bfloat16
        
    qwen = QwenEmbedder(dtype=dtype)  
    aligner = AudioEmbedder(device=device, dtype=dtype)

    proj_head = aligner.projection_head()
    proj_head.train()

    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))

    optimizer = torch.optim.AdamW(
        [{"params": proj_head.parameters()}, {"params": [logit_scale]}],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ======= RESUME SUPPORT =======
    start_epoch = 0
    latest_path = Path(CHECKPOINT_DIR) / "latest.pt"
    
    resume_file = os.environ.get("RESUME_PATH", str(latest_path))
    if Path(resume_file).exists():
        print(f"Loading checkpoint parameters from: {resume_file}")
        # weights_only=False because optimizer states contain objects
        checkpoint = torch.load(resume_file, map_location=device, weights_only=False)
        proj_head.load_state_dict(checkpoint["proj_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logit_scale.data = checkpoint["logit_scale"].to(device)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Successfully resumed at epoch {start_epoch}")
    else:
        print("No prior checkpoint found. Beginning new training run.")

    # Data Supply
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4 if torch.cuda.is_available() else 0))
    print(f"Configuring Dataloader with {num_workers} parallel workers.")
    dataset = AudioTextDataset(CSV_PATH, AUDIO_DIR)
    
    if len(dataset) == 0:
        print("CRITICAL: No valid dataset found to train on. Check directories. Exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    print(f"Starting Training Loop! Batches per epoch: {len(loader)}, Total samples: {len(dataset)}")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for audios, texts in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                text_embeds = qwen.embed_batch(texts).to(device, dtype=dtype)

                inputs = aligner._processor(
                    audio=audios,
                    sampling_rate=aligner.CLAP_SAMPLE_RATE, # 48000
                    return_tensors="pt",
                    padding=True,
                )
                input_features = inputs["input_features"].to(device, dtype=dtype)

                audio_outputs = aligner._audio_model(input_features=input_features)
                clap_embed = aligner._audio_projection(audio_outputs.pooler_output)
                clap_embed = F.normalize(clap_embed, p=2, dim=-1)

            # Conditional AMP
            amp_device = device.type
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                audio_embeds = proj_head(clap_embed.float())

                # Contrastive Loss (InfoNCE)
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
            pbar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{scale.item():.2f}")

        avg_loss = epoch_loss / len(loader)
        print(f"\n==== Epoch {epoch + 1} Summary ====")
        print(f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Temp: {scale.item():.2f}")

        scheduler.step()

        # ======= FULL STATE CHECKPOINT =======
        checkpoint = {
            "epoch": epoch,
            "proj_head_state_dict": proj_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "logit_scale": logit_scale.data,
        }
        
        ckpt_path = Path(CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        latest_path = Path(CHECKPOINT_DIR) / "latest.pt"
        
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, latest_path)
        
        print(f"-> Saved robust Checkpoints to {CHECKPOINT_DIR}\n")

if __name__ == "__main__":
    train()
