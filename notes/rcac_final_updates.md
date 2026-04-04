# Engine on Gautschi: Final Update Checklist

## 1) Make the training script save a real checkpoint, not only the projection head weights

Your current `train_loop.py` saves only:

```python
torch.save(proj_head.state_dict(), ckpt_path)
```

That is enough to recover the projection head weights, but not enough to resume training safely after an interruption.

### Update it to save a full training checkpoint

Save at least:

* `proj_head.state_dict()`
* `optimizer.state_dict()`
* `scheduler.state_dict()`
* `logit_scale`
* `epoch`
* optionally RNG states for reproducibility

### Why this matters on Gautschi

If you use `preemptible` jobs on the AI partition, your job may be interrupted. A full checkpoint lets you resume instead of restarting from scratch.

### Recommended checkpoint layout

Save checkpoints under scratch, for example:

```text
$RCAC_SCRATCH/engine-checkpoints/audio_projection/
```

### Recommended filename pattern

Use something like:

```text
checkpoint_epoch_001.pt
checkpoint_epoch_002.pt
latest.pt
```

Keep `latest.pt` updated each epoch so resuming is simple.

---

## 2) Add resume support to `train_loop.py`

Right now the training loop always starts from scratch.

### Update

Add support for:

* `RESUME_PATH` environment variable
* automatic loading of a previous checkpoint if it exists
* restoring the optimizer and scheduler states

### Why this matters

On RCAC, long jobs can be delayed, time out, or be preempted. Resume support is the difference between a resilient run and a wasted run.

---

## 3) Unify the audio sample rate across download and training

This is the clearest functional mismatch in the repo.

### Current state

* `fetch_yt_sample.py` downloads audio at **16 kHz mono**.
* `train_loop.py` reloads audio using `librosa.load(..., sr=48000, mono=True)`.
* `AudioEmbedder` is documented as using CLAP’s **48 kHz** path.

### Update

Pick one target sample rate and make all three places consistent:

* downloader
* training loader
* `AudioEmbedder` preprocessing expectation

### Recommended choice

Use **48 kHz** everywhere if you want to match CLAP’s native preprocessing path.

That means:

* change yt-dlp / ffmpeg trimming in `fetch_yt_sample.py` to export 48 kHz mono WAV
* keep `librosa.load(..., sr=48000, mono=True)` in training
* ensure `AudioEmbedder` uses the same expectation

### Why this matters

If the downloader writes 16 kHz and the trainer resamples to 48 kHz, your pipeline still works, but you add unnecessary conversion and extra overhead. Matching the sample rate end-to-end is cleaner and easier to debug.

---

## 4) Stop using silence as a hidden fallback for missing audio

In `train_loop.py`, missing clips return a second of zeros.

### Update

Replace the silent fallback with one of these policies:

* skip missing examples entirely
* pre-filter the CSV to only rows with valid files
* maintain a dedicated manifest of verified audio paths

### Why this matters

Silence is not a real training example. If missing clips are frequent, the model will learn from corrupted pairs.

### Best practice

For training, build a verified subset first and train only on that.

---

## 5) Filter or randomize the AudioSetCaps subset instead of always taking the first rows

`prepare_subset.py` currently does:

```python
subset = df.head(min(args.n_samples, len(df)))
```

### Update

Consider replacing the head-based selection with a random sample.

### Why this matters

The first rows may not be representative. A random subset is usually better for model training.

### Better behavior

Add a seed option and use:

```python
df.sample(n=..., random_state=seed)
```

This gives you reproducibility and a more balanced subset.

---

## 6) Make `prepare_subset.py` and `fetch_yt_sample.py` use explicit output locations

Right now both scripts write into relative `data/` paths.

### Update

Add CLI arguments or environment variable support for:

* output CSV location
* audio output directory
* cache directory if needed

### Why this matters on RCAC

You want large generated files in scratch, not in home.

### Recommended pattern

Use environment variables like:

* `CSV_PATH`
* `AUDIO_DIR`
* `CHECKPOINT_DIR`

or pass them explicitly on the command line.

---

## 7) Add a strict CSV schema check in `fetch_yt_sample.py`

The downloader assumes the CSV contains fields like `id`, `start_time`, and `end_time`.

### Update

Validate the dataframe columns before download begins.

### Why this matters

If the schema changes or a column is missing, the script should fail early and clearly instead of silently using fallback values.

### Recommended checks

Verify at least:

* `id`
* the timestamp columns you really depend on
* `caption`

If the actual AudioSetCaps schema uses different column names, update the script to match the real names exactly.

---

## 8) Normalize all IDs and paths in the training and download scripts

### Current issue

`yt_id` is sometimes treated as a string and sometimes not.

### Update

Force the ID to string before prefix stripping and file path generation.

### Why this matters

It avoids weird edge cases when pandas reads IDs as numeric-like or mixed types.

---

## 9) Make the training script print clearer progress and save a `latest.pt`

### Update

At the end of each epoch, save:

* `checkpoint_epoch_###.pt`
* `latest.pt`

Also print:

* epoch number
* average loss
* current learning rate
* current temperature
* checkpoint path

### Why this matters

If a job fails halfway through, you immediately know the last good checkpoint.

---

## 10) Make the training loop explicitly consume the model and dataset paths you intend to use on RCAC

### Current state

The script defaults to:

* `data/AudioSetCaps_caption_subset.csv`
* `data/audio`
* `data/checkpoints`

### Update

On RCAC, override these defaults in the job script with scratch-based paths.

### Recommended values

* `CSV_PATH=$HOME/engine/data/AudioSetCaps_caption_subset.csv` or a scratch manifest if you generate one there
* `AUDIO_DIR=$RCAC_SCRATCH/engine/audio`
* `CHECKPOINT_DIR=$RCAC_SCRATCH/engine/checkpoints`

The CSV can live in home if it is small. The audio and checkpoints should live in scratch.

---

## 11) Make the repo import path less brittle if possible

`train_loop.py` currently does this:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
```

### Update

This works only if the file stays at the same depth.

### Better options

* package the project cleanly so `src` can be imported normally
* run from repo root with the right `PYTHONPATH`
* or make a clearer project-root helper

### Why this matters

It reduces the chance of broken imports when you move files or change launch paths.

---

## 12) Confirm the audio embedder and the training loop are using the same CLAP preprocessing assumptions

Your docs say:

* `AudioEmbedder` loads CLAP
* CLAP uses frozen audio encoder + audio projection
* the trainable projection head maps to Qwen space

### Update

Double-check these details line up in code:

* CLAP audio model is frozen in both training and inference
* the same `CLAP_SAMPLE_RATE` is used in download, preprocessing, and training
* the projection head input size is really the CLAP output size you think it is
* the output size is 2048 everywhere

### Why this matters

This is the core bridge of the whole multimodal system.

---

## 13) Make the Qwen and audio embedding dimensions explicit in one place

### Update

Keep a single source of truth for:

* embedding dimension = 2048
* Qdrant vector size = 2048
* projection head output size = 2048

### Why this matters

Your indexer, training code, and Qdrant schema must all agree.

---

## 14) Make the indexer and the training pipeline agree about file layouts

Your repo structure is good, but make sure the run-time assumptions are explicit:

* `trusted/` is for indexable user files
* `data/` is for training artifacts and should be gitignored
* `models/` stores the pre-downloaded Qwen weights
* `tests/` contains fixtures only

### Update

Document these rules in `README.md` and keep the scripts aligned with them.

---

# Final recommended order of work

1. Update the training script to save full checkpoints and support resume.
2. Unify the audio sample rate across download and training.
3. Move generated audio and checkpoints to scratch paths.
4. Verify the CSV schema and sample selection logic.
5. Run a small preprocessing job.
6. Run a short training job.
7. Confirm checkpoint loading works.
8. Launch the full training run on Gautschi.

Yes, this repo is structurally aligned with the RCAC plan, but it needs a few important updates before you should trust it for long training runs:

* better checkpointing
* resume support
* one consistent audio sample rate
* scratch-based storage for large artifacts
* stricter CSV and audio-file validation
* an RCAC job script that activates the environment and sets the correct paths
