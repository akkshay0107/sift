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
export BATCH_SIZE=2048

# Keep high epochs
export EPOCHS=500

# ---------------------------------------------------------
# REGULARIZATION TWEAKS (COMBATING OVERFITTING)
# ---------------------------------------------------------
# 1. Lower LR to let the model settle gracefully
export LR=3e-4

export WEIGHT_DECAY=0.05

# 3. Increase learning rate warmup/decay stability via Gradient Clipping
export GRAD_CLIP_NORM=0.5

uv run python -m src.embed.train.train_loop
