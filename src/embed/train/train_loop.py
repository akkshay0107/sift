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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.embed.audio import AudioEmbedder
from src.embed.qwen import QwenEmbedder

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_PATH = "data/AudioSetCaps_caption_subset.csv"
AUDIO_DIR = "data/audio"
CHECKPOINT_DIR = "data/checkpoints"
BATCH_SIZE = 4  # Start small; on H100, increase dramatically to 128/256
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.2


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
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
        
        # Load audio (fallback to silence if missing to keep batching intact)
        if audio_path.exists():
            audio_array, _ = librosa.load(audio_path, sr=48000, mono=True)
        else:
            audio_array = np.zeros(48000, dtype=np.float32)
            
        # Dynamically fetch caption (support different valid headers)
        caption = "Missing caption"
        for col in ["caption", "Captions", "text"]:
            if col in row:
                caption = str(row[col])
                break
                
        return audio_array, caption


def collate_fn(batch):
    audios, texts = zip(*batch)
    return list(audios), list(texts)


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Pipeline Embedders (loads weights)
    print("Loading Base Models...")
    qwen = QwenEmbedder(dtype=torch.bfloat16)  # Generates text targets
    aligner = AudioEmbedder(device=device, dtype=torch.bfloat16)
    
    # 2. Extract and Isolate the Trainable Projection Head
    proj_head = aligner.projection_head()
    proj_head.train()
    
    # 3. Learnable contrastive temperature scalar
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    
    optimizer = torch.optim.AdamW([
        {"params": proj_head.parameters()},
        {"params": [logit_scale]}
    ], lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 4. Data Supply
    dataset = AudioTextDataset(CSV_PATH, AUDIO_DIR)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Starting Training! Batches per epoch: {len(loader)}")
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (audios, texts) in enumerate(pbar):
            optimizer.zero_grad()
            
            # --- FEATURE EXTRACTION (FROZEN BACKBONES) ---
            with torch.no_grad():
                # Text Targets
                # Returns shape [B, 2048] CPU float32 -> Move to GPU bfloat16
                text_embeds = qwen.embed_batch(texts).to(device, dtype=torch.bfloat16)
                
                # Audio Encoding via CLAP
                inputs = aligner._processor(
                    audio=audios,
                    sampling_rate=aligner.CLAP_SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                input_features = inputs["input_features"].to(device, dtype=torch.bfloat16)
                
                # Passes through CLAP backbone, skipping its default projection head
                audio_outputs = aligner._audio_model(input_features=input_features)
                clap_embed = aligner._audio_projection(audio_outputs.pooler_output)
                clap_embed = F.normalize(clap_embed, p=2, dim=-1) # Shape [B, 512]
            
            # --- TRAINING PROJECTION HEAD ---
            # Cast using AMP
            device_type_str = "cuda" if "cuda" in str(device) else "cpu"
            with torch.amp.autocast(device_type=device_type_str, dtype=torch.bfloat16):
                
                # The projection head natively runs with float32 internal activations, but amp casts it
                audio_embeds = proj_head(clap_embed.float()) # Shape [B, 2048], L2 normalized natively
                
                # Contrastive (Symmetric InfoNCE) Loss
                scale = torch.clamp(logit_scale.exp(), max=100.0)
                logits_per_audio = scale * audio_embeds @ text_embeds.t()
                logits_per_text = logits_per_audio.t()
                
                # Diagonal pairs are the matching ground truth
                labels = torch.arange(len(audio_embeds), device=device)
                
                loss_a = F.cross_entropy(logits_per_audio, labels)
                loss_t = F.cross_entropy(logits_per_text, labels)
                loss = (loss_a + loss_t) / 2.0
                
            # --- BACKPROP ---
            # No GradScaler needed due to bfloat16 dynamics
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{scale.item():.2f}")
            
        # --- CHECKPOINT ---
        ckpt_path = Path(CHECKPOINT_DIR) / f"proj_head_ep{epoch+1}.pt"
        torch.save(proj_head.state_dict(), ckpt_path)
        print(f"-> Saved Checkpoint to {ckpt_path}")
        

if __name__ == "__main__":
    train()
