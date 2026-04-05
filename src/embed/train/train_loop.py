import math
import os
import sys
import gc
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.embed.audio import AudioEmbedder, ProjectionHead
from src.embed.qwen import QwenEmbedder

scratch_base = os.environ.get("RCAC_SCRATCH", "./scratch")
CSV_PATH = os.environ.get("CSV_PATH", str(Path(scratch_base) / "engine" / "data" / "AudioSetCaps_caption_subset.csv"))
AUDIO_DIR = os.environ.get("AUDIO_DIR", str(Path(scratch_base) / "engine" / "audio"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", str(Path(scratch_base) / "engine" / "checkpoints"))

PRECOMPUTE_BATCH_SIZE = int(os.environ.get("PRECOMPUTE_BATCH_SIZE", 8)) # Small batches for heavy models
TRAIN_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1024)) # Massive batches for gradient optimization
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


def load_dataset() -> AudioTextDataset:
    """Step 1: Parse the CSV and load the valid subset into memory."""
    print("Configuring Dataloader...")
    dataset = AudioTextDataset(CSV_PATH, AUDIO_DIR)
    
    if len(dataset) == 0:
        raise ValueError("CRITICAL: No valid dataset found to train on. Check directories. Exiting.")
        
    return dataset


def precompute_embeddings(dataset: AudioTextDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, ProjectionHead]:
    """Step 2: Calculate embedding maps and wipe the foundation models from memory."""
    print("Initializing Base Models...")
    
    qwen = QwenEmbedder()  
    aligner = AudioEmbedder(device=device)

    # Grab the Qwen dtype to use for the text/audio caching arrays later
    cache_dtype = qwen._embedder.model.dtype 
    
    # Detach projection head to survive model deletion
    proj_head = aligner.projection_head().float()
    
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4 if torch.cuda.is_available() else 0))
    pre_loader = DataLoader(
        dataset,
        batch_size=PRECOMPUTE_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    all_text_embeds = []
    all_clap_embeds = []

    print("Precomputing Base Embeddings (This runs exactly ONCE)...")
    for audios, texts in pre_loader:
        with torch.no_grad():
            text_emb = qwen.embed_batch(texts).to(device, dtype=cache_dtype)

            inputs = aligner._processor(
                audio=audios,
                sampling_rate=aligner.CLAP_SAMPLE_RATE, # 48000
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs["input_features"].to(device, dtype=aligner.dtype)

            audio_outputs = aligner._audio_model(input_features=input_features)
            clap_emb = aligner._audio_projection(audio_outputs.pooler_output)
            clap_emb = F.normalize(clap_emb, p=2, dim=-1)

            # Move to CPU RAM temporarily to save precious GPU VRAM
            all_text_embeds.append(text_emb.cpu())
            all_clap_embeds.append(clap_emb.cpu())
            
    tensor_text = torch.cat(all_text_embeds, dim=0)
    tensor_clap = torch.cat(all_clap_embeds, dim=0)
    
    print(f"Precomputed Shapes - Audio: {tensor_clap.shape}, Text: {tensor_text.shape}")
    
    # Memory Cleanup: Free the massive models before the training loop
    print("Precomputing complete! Purging Foundation Models from memory...")
    del qwen
    del aligner._audio_model
    del aligner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return tensor_clap, tensor_text, proj_head


def train_projection_head(tensor_clap: torch.Tensor, tensor_text: torch.Tensor, proj_head: ProjectionHead, device: torch.device):
    """Step 3: Train the Projection Head natively against the cached tensors."""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be managed in: {CHECKPOINT_DIR}")

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

    fast_dataset = TensorDataset(tensor_clap, tensor_text)
    fast_loader = DataLoader(fast_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    print(f"Starting FAST Training Loop! Batches per epoch: {len(fast_loader)}")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        
        for batch_clap, batch_text in fast_loader:
            # Move precomputed batch directly to GPU using its cached datatype
            batch_clap = batch_clap.to(device)
            batch_text = batch_text.to(device)
            
            optimizer.zero_grad(set_to_none=True)

            # Pass directly through our untended projection head
            raw_audio_embeds = proj_head(batch_clap.float())

            # CRITICAL: L2 Normalize and safely cast to float32 to ensure true Cosine Similarity for InfoNCE
            audio_embeds = F.normalize(raw_audio_embeds, p=2, dim=-1).float()
            batch_text = F.normalize(batch_text, p=2, dim=-1).float()

            # Contrastive Loss (InfoNCE)
            scale = torch.clamp(logit_scale.exp(), max=100.0)
            logits_per_audio = scale * audio_embeds @ batch_text.t()
            logits_per_text = logits_per_audio.t()

            labels = torch.arange(len(audio_embeds), device=device)
            loss_a = F.cross_entropy(logits_per_audio, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_a + loss_t) / 2.0

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(fast_loader)
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


def run_pipeline():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using compute device: {device}")
    
    # 1. Parse and extract validated dataset mapping (CPU bound)
    dataset = load_dataset()
    
    # 2. Extract arrays and wipe foundational memory limits (Heavy inference mapping)
    tensor_clap, tensor_text, proj_head = precompute_embeddings(dataset, device)
    
    # 3. High Velocity gradient matching (Lightweight loop computing)
    train_projection_head(tensor_clap, tensor_text, proj_head, device)


if __name__ == "__main__":
    run_pipeline()
