import gc
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.embed.audio import AudioEmbedder, ProjectionHead
from src.embed.qwen import QwenEmbedder

scratch_base = os.environ.get("RCAC_SCRATCH", "./scratch")
CSV_PATH = os.environ.get(
    "CSV_PATH",
    str(Path(scratch_base) / "engine" / "data" / "AudioSetCaps_caption_subset.csv"),
)
AUDIO_DIR = os.environ.get("AUDIO_DIR", str(Path(scratch_base) / "engine" / "audio"))
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR", str(Path(scratch_base) / "engine" / "checkpoints")
)

PRECOMPUTE_BATCH_SIZE = int(os.environ.get("PRECOMPUTE_BATCH_SIZE", 8))
TRAIN_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1024))
EPOCHS = int(os.environ.get("EPOCHS", 25))
LR = float(os.environ.get("LR", 2.5e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.1))

VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.1))
SEED = int(os.environ.get("SEED", 42))
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

# Temperature is inverse scale: scale = exp(logit_scale) = 1 / temp
MIN_TEMP = float(os.environ.get("MIN_TEMP", 0.01))
MAX_TEMP = float(os.environ.get("MAX_TEMP", 0.20))


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
            audio_array, _ = librosa.load(audio_path, sr=48000, mono=True)
        except Exception as e:
            print(f"Warning: Failed to load {audio_path}: {e}")
            audio_array = np.zeros(48000, dtype=np.float32)

        return audio_array, str(row["caption"])


def collate_fn(batch):
    audios, texts = zip(*batch)
    return list(audios), list(texts)


def load_dataset() -> AudioTextDataset:
    print("Configuring Dataloader...")
    dataset = AudioTextDataset(CSV_PATH, AUDIO_DIR)

    if len(dataset) == 0:
        raise ValueError(
            "CRITICAL: No valid dataset found to train on. Check directories. Exiting."
        )

    return dataset


def precompute_embeddings(
    dataset: AudioTextDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, ProjectionHead]:
    print("Initializing Base Models...")

    qwen = QwenEmbedder()
    aligner = AudioEmbedder(device=device)

    cache_dtype = qwen._embedder.model.dtype
    proj_head = aligner.projection_head().float()

    num_workers = int(
        os.environ.get("SLURM_CPUS_PER_TASK", 4 if torch.cuda.is_available() else 0)
    )
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

    print("Precomputing Base Embeddings...")
    for audios, texts in pre_loader:
        with torch.no_grad():
            text_emb = qwen.embed_batch(texts).to(device, dtype=cache_dtype)

            inputs = aligner._processor(
                audio=audios,
                sampling_rate=aligner.CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs["input_features"].to(device, dtype=aligner.dtype)

            audio_outputs = aligner._audio_model(input_features=input_features)
            clap_emb = aligner._audio_projection(audio_outputs.pooler_output)
            clap_emb = F.normalize(clap_emb, p=2, dim=-1)

            all_text_embeds.append(text_emb.cpu())
            all_clap_embeds.append(clap_emb.cpu())

    tensor_text = torch.cat(all_text_embeds, dim=0)
    tensor_clap = torch.cat(all_clap_embeds, dim=0)

    print(f"Precomputed Shapes - Audio: {tensor_clap.shape}, Text: {tensor_text.shape}")

    print("Precomputing complete! Purging Foundation Models from memory...")
    del qwen
    del aligner._audio_model
    del aligner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tensor_clap, tensor_text, proj_head


def split_cached_tensors(
    tensor_clap: torch.Tensor,
    tensor_text: torch.Tensor,
    val_split: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    num_samples = tensor_clap.size(0)
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to create a train/val split.")

    if not (0.0 < val_split < 0.5):
        raise ValueError(f"VAL_SPLIT must be in (0, 0.5), got {val_split}")

    num_val = max(1, int(num_samples * val_split))
    num_train = num_samples - num_val
    if num_train < 1:
        raise ValueError("Validation split left no samples for training.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]

    train_dataset = TensorDataset(tensor_clap[train_idx], tensor_text[train_idx])
    val_dataset = TensorDataset(tensor_clap[val_idx], tensor_text[val_idx])

    print(f"Train/Val split: {len(train_dataset)} / {len(val_dataset)}")
    return train_dataset, val_dataset


def clamp_logit_scale_(logit_scale: torch.Tensor, min_temp: float, max_temp: float):
    if min_temp <= 0 or max_temp <= 0 or min_temp > max_temp:
        raise ValueError(f"Invalid temperature bounds: min={min_temp}, max={max_temp}")

    min_logit = math.log(1.0 / max_temp)
    max_logit = math.log(1.0 / min_temp)
    with torch.no_grad():
        logit_scale.clamp_(min=min_logit, max=max_logit)


def compute_contrastive_loss(
    batch_clap: torch.Tensor,
    batch_text: torch.Tensor,
    proj_head: ProjectionHead,
    logit_scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    batch_clap = batch_clap.to(device, non_blocking=True)
    batch_text = batch_text.to(device, non_blocking=True)

    raw_audio_embeds = proj_head(batch_clap.float())
    audio_embeds = F.normalize(raw_audio_embeds, p=2, dim=-1).float()
    text_embeds = F.normalize(batch_text.float(), p=2, dim=-1).float()

    scale = logit_scale.exp()
    logits_per_audio = scale * (audio_embeds @ text_embeds.t())
    logits_per_text = logits_per_audio.t()

    labels = torch.arange(audio_embeds.size(0), device=device)
    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = 0.5 * (loss_a + loss_t)

    return loss, float(scale.detach().item())


@torch.no_grad()
def evaluate(
    proj_head: ProjectionHead,
    logit_scale: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    proj_head.eval()
    total_loss = 0.0
    total_batches = 0
    last_scale = float(logit_scale.exp().detach().item())

    for batch_idx, (batch_clap, batch_text) in enumerate(loader):
        loss, last_scale = compute_contrastive_loss(
            batch_clap, batch_text, proj_head, logit_scale, device
        )
        total_loss += loss.item()
        total_batches += 1

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, last_scale


def train_projection_head(
    tensor_clap: torch.Tensor,
    tensor_text: torch.Tensor,
    proj_head: ProjectionHead,
    device: torch.device,
):
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be managed in: {CHECKPOINT_DIR}")

    train_dataset, val_dataset = split_cached_tensors(
        tensor_clap, tensor_text, VAL_SPLIT, SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    proj_head = proj_head.to(device).float()

    # Initialize projection head with Xavier uniform
    for m in proj_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    proj_head.train()

    # RESTORED LEARNABLE TEMPERATURE (Starting at 0.07)
    default_temp = 0.07
    logit_scale = nn.Parameter(
        torch.tensor(math.log(1.0 / default_temp), device=device, dtype=torch.float32)
    )
    clamp_logit_scale_(logit_scale, MIN_TEMP, MAX_TEMP)

    optimizer = torch.optim.AdamW(
        [{"params": proj_head.parameters()}, {"params": [logit_scale]}],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=1e-6,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience = 10
    epochs_no_improve = 0

    latest_path = Path(CHECKPOINT_DIR) / "latest.pt"
    best_path = Path(CHECKPOINT_DIR) / "best.pt"
    resume_file = os.environ.get("RESUME_PATH", str(latest_path))

    if Path(resume_file).exists():
        print(f"Loading checkpoint parameters from: {resume_file}")
        checkpoint = torch.load(resume_file, map_location=device, weights_only=False)

        proj_head.load_state_dict(checkpoint["proj_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        with torch.no_grad():
            if "logit_scale" in checkpoint:
                if isinstance(checkpoint["logit_scale"], torch.Tensor):
                    logit_scale.copy_(checkpoint["logit_scale"].to(device))
                else:
                    logit_scale.copy_(
                        torch.tensor(checkpoint["logit_scale"], device=device)
                    )
        clamp_logit_scale_(logit_scale, MIN_TEMP, MAX_TEMP)

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Successfully resumed at epoch {start_epoch}")
    else:
        print("No prior checkpoint found. Beginning new training run.")

    print(
        f"Starting training | train batches/epoch: {len(train_loader)} | "
        f"val batches: {len(val_loader)}"
    )

    for epoch in range(start_epoch, EPOCHS):
        proj_head.train()
        epoch_train_loss = 0.0

        for batch_clap, batch_text in train_loader:
            optimizer.zero_grad(set_to_none=True)

            loss, _ = compute_contrastive_loss(
                batch_clap, batch_text, proj_head, logit_scale, device
            )

            loss.backward()
            nn.utils.clip_grad_norm_(proj_head.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            clamp_logit_scale_(logit_scale, MIN_TEMP, MAX_TEMP)

            epoch_train_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        avg_val_loss, val_scale = evaluate(proj_head, logit_scale, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        current_temp = 1.0 / val_scale

        print(f"\n==== Epoch {epoch + 1} Summary ====")
        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Scale: {val_scale:.2f} | "
            f"Temp: {current_temp:.4f}"
        )

        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "proj_head_state_dict": proj_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "logit_scale": logit_scale.detach().clone(),
            "config": {
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "val_split": VAL_SPLIT,
                "min_temp": MIN_TEMP,
                "max_temp": MAX_TEMP,
                "seed": SEED,
            },
        }

        ckpt_path = Path(CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, ckpt_path)

        torch.save(checkpoint, latest_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)
            print(f"-> New best checkpoint saved: {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"-> Early stopping triggered at epoch {epoch + 1}")
                break

        print(
            f"-> Saved latest checkpoint to {CHECKPOINT_DIR} (Epoch backups every 10)\n"
        )


def run_pipeline():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using compute device: {device}")

    dataset = load_dataset()
    tensor_clap, tensor_text, proj_head = precompute_embeddings(dataset, device)
    train_projection_head(tensor_clap, tensor_text, proj_head, device)


if __name__ == "__main__":
    run_pipeline()
