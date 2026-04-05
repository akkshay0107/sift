# Sift: Multimodal Memory Engine

Sift is a high-performance local embedding and retrieval engine designed for instant multimodal semantic search across text, images, audio, and video. It operates as a personal memory layer, providing zero-friction access to indexed data through a unified vector space.

---

## Core Architecture

### 1. Multimodal Core (Qwen3-VL)

The backbone of Sift is **Qwen3-VL-Embedding-2B**, which natively handles text, images, and video.

- **Unified Vector Space**: All modalities are mapped into a shared 2048-dimensional space.
- **Normalization**: All output vectors are L2-normalized float32 representations.
- **Backbone**: Leverages specialized vision-text transformer blocks for deep semantic understanding.

### 2. Audio Bridge (CLAP + Projection)

Unified search for audio is achieved through a CLAP-to-Qwen adapter:

- **Audio Backbone**: `laion/clap-htsat-unfused` (frozen).
- **Projection Head**: A learned 2-layer MLP (512 -> 1024 -> 2048) that aligns CLAP's audio embeddings with Qwen's vision-text space.
- **Training**: Optimized using Contrastive InfoNCE loss on the AudioSetCaps dataset to bridge the gap between audio features and textual/visual concepts.

### 3. Processing Pipelines

- **OCR Chain**: Uses EasyOCR to extract text from images, which is then embedded via Qwen for semantic search.
- **Transcription Chain**: Uses Faster-Whisper (large-v3) to generate transcripts from audio, which are also embedded via Qwen to provide text-based retrieval of audio segments.
- **Metadata Chain**: Generates text-based metadata summaries (filename, extension, modality, and timestamps) for every file to ensure discoverability even without deep content-based matches.

---

## Indexing System

Sift uses an incremental indexing strategy to minimize redundant processing and maximize performance:

- **Change Detection**: Uses BLAKE3 hashing to track file modifications and skip unchanged files.
- **Vector Database**: Integrated with Qdrant for high-speed similarity search and persistent storage.
- **Pipelines**: Files are automatically routed to appropriate processing chains based on MIME type and file extension.
- **Source Mapping**: Maintains strict mapping between source paths and vector IDs to allow for re-indexing and deletion.

---

## Search Capabilities

- **Multimodal Retrieval**: Search by natural language to find related text, images, audio recordings, or video clips in a single query.
- **Result Bundling**: Groups similar snippets or related files using a hybrid scoring system that considers embedding similarity, temporal proximity, and filename Jaccard similarity.
- **Score Thresholding**: Configurable thresholds allow for filtering low-relevance matches to maintain high precision in large datasets.

---

## User Interface

The project includes a futuristic, minimalist desktop application built with PySide6:

- **Categorized Results**: Specialized views for file lists, top matches, live metadata snapshots, and recognized entities.
- **Keyboard-First Design**: Optimized for rapid use with shortcuts like `Esc` to close and `Enter` to execute queries.

---

## Setup & Installation

### 1. Prerequisites

- Python 3.12 (managed via `uv`)
- Docker (required for running the Qdrant vector database)

### 2. Environment Setup

Clone the repository and install dependencies using `uv`:

```bash
uv sync
```

### 3. Model Weights

Download the required pre-trained weights into the `models/` directory:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-VL-Embedding-2B', local_dir='models/Qwen3-VL-Embedding-2B')"
```

### 4. Qdrant Deployment

Run the Qdrant vector database locally using Docker:

```bash
docker pull qdrant/qdrant
mkdir -p qdrant_storage
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

Alternatively, use podman based on your system preference.

### 5. System Dependencies (Linux)

Install the following libraries to support the Qt-based UI on Linux systems:

```bash
sudo apt update
sudo apt install -y libgl1 libegl1 libdbus-1-3 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
  libxcb-xinerama0 libx11-xcb1 libxrender1 libxi6 libxcomposite1 libxtst6
```

---

## Usage

### Indexing Files

To perform a one-time indexing of files in your configured directories:

```bash
uv run python -m src.indexer.run_indexer
```

### Unified Daemon

To run the main daemon that performs the startup scan, watches for new/modified files, and keeps the desktop UI resident in a hidden state:

```bash
uv run python -m src.daemon
```

To run the daemon persistently in the background (even after closing your terminal), use `nohup`:

```bash
nohup uv run python -m src.daemon > daemon.log 2>&1 &
```

If you only want the legacy indexing-only watcher without the UI process attached:

```bash
uv run python -m src.indexer.daemon
```

### Running Search

To launch the interactive desktop application:

```bash
uv run python main.py
```

To run a simple CLI-based search:

```bash
uv run python main.py --cli
```

---

## Training on RCAC Gautschi

Sift includes specialized scripts for training and extending the audio alignment layer on high-performance computing clusters:

1.  **Prepare Subset**: `uv run python -m src.embed.train.prepare_subset` (Prepares training data).
2.  **Fetch Audio**: `uv run python -m src.embed.train.fetch_yt_sample` (Downloads training samples).
3.  **Train Loop**: `uv run python -m src.embed.train.train_loop` (Starts the alignment training).
4.  **Checkpointing**: Training automatically saves to `latest.pt` and supports seamless resumption after preemption.

---

## Project Structure

```
src/
  embed/        # Model wrappers (Qwen, CLAP, Whisper, EasyOCR)
  indexer/      # File monitoring, hashing, and Qdrant logic
  search/       # Retrieval, ranking, and temporal bundling logic
  ui/           # PySide6 desktop application implementation
trusted/        # Default user data directory for indexing
models/         # Local directory for model weights and checkpoints
notes/          # Technical specifications, schemas, and runbooks
```
