# RCAC Gautschi Runbook (Copy & Paste Edition)

Since you are using `uv`, you don't need to rely on RCAC's heavy, pre-packaged Anacondas. You can install `uv` directly inside your user profile.

Follow these exact copy-paste blocks. **You do not need to open any text editors.**

---

### Step 1: Initial Setup (Run on Gautschi)
Paste this entire block into your Gautschi terminal. It loads Python, clones your code, installs `uv`, and synchronizes your dependencies.

```bash
# 1. Load the officially supported Python module
module load python/3.11.9

# 2. Clone your code and enter the directory
git clone https://github.com/akkshay0107/engine.git
cd engine

# 3. Install the 'uv' package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Append uv to your path so the terminal recognizes the command
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 5. Lock and install your exact environment
uv sync
```

---

### Step 2: Transfer Qwen Models (Run on your MAC - NOT Gautschi)
Open a **brand new Terminal tab on your MacBook** so you are operating locally. Ensure you are inside your `Desktop/engine` folder. 

Run this command to push the massive model weights straight to Gautschi:
*(Type your password when prompted)*

```bash
rsync -avz --progress models/Qwen3-VL-Embedding-2B sriram54@gautschi.rcac.purdue.edu:~/engine/models/
```

---

### Step 3: Create and Run the "Prep Ticket" (Run on Gautschi)
Now that the code is ready, switch back to the **Gautschi terminal**. 
We need to download the 5000 YouTube clips. Paste this massive block to automatically generate `prep.slurm` and submit it to the `cpu` queue.

*(Note: Replace `YOUR_ACCOUNT_HERE` with your actual allocation name before running. If you don't know it, run the command `slist` to see your queues).*

```bash
cat << 'EOF' > prep.slurm
#!/bin/bash
#SBATCH --job-name=audio_prep
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --output=prep.out

cd $SLURM_SUBMIT_DIR
module load python/3.11.9
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1. Grab dataset (Full 5000 samples)
uv run python -m src.embed.train.prepare_subset --n_samples 5000

# 2. Download the audio locally to the massive scratch drive
uv run python -m src.embed.train.fetch_yt_sample

# 3. Quick 2-epoch dry run to test paths
export HF_HOME="$RCAC_SCRATCH/huggingface_cache"
BATCH_SIZE=2 EPOCHS=2 uv run python -m src.embed.train.train_loop
EOF

# Submit the prep job!
sbatch prep.slurm
```

You can watch it run with `squeue -u sriram54`. Once it disappears from the queue, type `cat prep.out` to see the logs. It should have downloaded everything and done a quick test run.

---

### Step 4: Create and Run the "Train Ticket" (Run on Gautschi)
If Step 3 went smoothly, it's time to leverage the massive H100 GPUs. 
Paste this block into Gautschi to create `train.slurm` and launch the final training loop.

```bash
cat << 'EOF' > train.slurm
#!/bin/bash
#SBATCH --job-name=engine_train
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=train.out

cd $SLURM_SUBMIT_DIR
module load python/3.11.9
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Setup Hyperparameters and bypass HF Home quota
export BATCH_SIZE=128
export EPOCHS=50
export HF_HOME="$RCAC_SCRATCH/huggingface_cache"

# Train the model
uv run python -m src.embed.train.train_loop
EOF

# Submit the multi-hour GPU training job!
sbatch train.slurm
```

### End of Workflow
- Type `squeue -u sriram54` to monitor it.
- Your final weights will progressively serialize into `$RCAC_SCRATCH/engine/checkpoints/latest.pt`
- To review the training logs as it runs, type `cat train.out`.
