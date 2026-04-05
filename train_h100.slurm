#!/bin/bash
#SBATCH --job-name=engine_train
#SBATCH --account=mlp
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out

cd $SLURM_SUBMIT_DIR
module load python/3.11.9
export PATH="$HOME/.deno/bin:$HOME/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

export HF_HOME="$RCAC_SCRATCH/huggingface_cache"
export PYTHONUNBUFFERED=1

export CSV_PATH="$RCAC_SCRATCH/engine/data/AudioSetCaps_caption_subset.csv"

export PRECOMPUTE_BATCH_SIZE=16
export BATCH_SIZE=256

# Transferred to 60 epochs for 2 full cycles with early stopping
export EPOCHS=30

# ---------------------------------------------------------
# REGULARIZATION TWEAKS (COMBATING OVERFITTING)
# ---------------------------------------------------------
# 1. Lower LR to let the model settle gracefully (1e-4 for smaller batches)
export LR=5e-5

export WEIGHT_DECAY=0.15

# 3. Increase learning rate warmup/decay stability via Gradient Clipping
export GRAD_CLIP_NORM=0.5

uv run python -m src.embed.train.train_loop
