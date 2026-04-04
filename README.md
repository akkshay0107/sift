# Multimodal Memory Engine (MME)

A high-performance local embedding and retrieval engine designed for instant multimodal semantic search across Text, Images, and Audio. Built to operate as a "Memory Engine" with zero-friction access to your personal or project data.

## 🚀 Key Features

- **Unified Embedding Space**: Maps Text, Images, and Audio into a shared 2048-dimensional vector space.
- **Qwen3-VL Backbone**: Leverages the state-of-the-art **Qwen3-VL-Embedding-2B** for native vision and text understanding.
- **Audio-to-Multimodal Alignment**: Bridges the gap between Audio (CLAP) and Text/Vision (Qwen) via a learned Projection Head trained on AudioSetCaps.
- **OCR & Transcript Chains**: 
    - **OCR**: Extract and embed text from complex images using EasyOCR + Qwen.
    - **Transcription**: Convert audio to text via Faster-Whisper and embed the resulting transcript.
- **Hybrid Retrieval**: Search by raw content (Image/Audio) or by semantic descriptions (Text).
- **HPC Optimized**: Support for distributed training on RCAC Gautschi (H100/A100) with full checkpointing and resume capabilities.

---

## 🛠️ Technical Architecture

### 1. Multimodal Core (Qwen3-VL)
The engine uses **Qwen3-VL-Embedding-2B** as its primary anchor. 
- **Text/Images**: Handled natively by the Qwen transformer. 
- **Dimensionality**: All outputs are L2-normalized float32 vectors of size **2048**.

### 2. Audio Bridge (CLAP + Projection)
Since Qwen3-VL does not natively support audio, we implement a **CLAP-to-Qwen Adapter**:
- **Backbone**: `laion/clap-htsat-unfused` (frozen).
- **Projection Head**: A 2-layer MLP (512 → 1024 → 2048) that translates CLAP's audio features into the Qwen anchor space.
- **Training**: Trained using a Contrastive InfoNCE loss (Cosine Similarity) on the AudioSetCaps dataset.

### 3. Processing Chains
- **OCR Pipeline**: `EasyOCR` -> `Qwen Text Embedder`. Best for documents and screenshots.
- **Whisper Pipeline**: `Faster-Whisper` (large-v3) -> `Qwen Text Embedder`. Generates semantic descriptors for audio files.

---

## 📦 Setup & Installation

### 1. Environment
Ensure you have `uv` installed, then synchronize the environment:
```bash
uv sync
```

### 2. Model Downloads
Fetch the Qwen parameters (~5GB):
```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-VL-Embedding-2B', local_dir='models/Qwen3-VL-Embedding-2B')"
```

---

## 🔍 Usage Examples

### Semantic File Search
The main entry point (`main.py`) provides an interactive CLI to search through your `trusted/` directory.
```bash
uv run main.py
```

### Programmatic Embedding
```python
from src.embed.qwen import QwenEmbedder
from src.embed.audio import AudioEmbedder

# Initialize
qwen = QwenEmbedder()
audio_aligner = AudioEmbedder(projection_path="path/to/projection.pt")

# Cross-modal Similarity
text_vec = qwen.embed("Heavy rain on a metal roof")
audio_vec = audio_aligner.embed(waveform, sample_rate=48000)

similarity = (text_vec @ audio_vec.T).item()
```

---

## 🏗️ Indexing System
The indexer monitors the `trusted/` directory and synchronizes it with a local **Qdrant** vector database.

- **Incremental Updates**: Uses BLAKE3 content hashing to skip unchanged files.
- **Multi-Pipeline Expansion**: A single image is indexed as both a raw visual embedding AND an OCR text embedding. Audio is indexed as both a CLAP vector and a Whisper transcript.
- **Storage**: Qdrant runs locally (default port 6333).

To run the indexer manually:
```bash
uv run python -m src.indexer.run_indexer
```

---

## 🎓 Training (RCAC Gautschi)
The project includes a robust training pipeline for the Audio Projection Head, optimized for the RCAC Gautschi HPC cluster.

### 1. Prepare Dataset
```bash
# Fetches 5000 samples and formats for training
uv run python -m src.embed.train.prepare_subset --n_samples 5000
```

### 2. Fetch Audio (Rate-Limit Aware)
Downloads YouTube samples with anti-throttling logic and cookie support:
```bash
uv run python -m src.embed.train.fetch_yt_sample --cookies cookies.txt
```

### 3. Execute Training
Uses a "Fast Loop" strategy: precomputes frozen backbones to RAM/Disk once, then trains the MLP at 1000+ iterations/sec on H100 GPUs.
```bash
# Resumes automatically from latest.pt if found
uv run python -m src.embed.train.train_loop
```

---

## 📂 Project Structure
- `src/embed/`: Core model wrappers (Qwen, CLAP, Whisper).
- `src/indexer/`: File monitoring, Qdrant integration, and indexing pipelines.
- `src/search/`: Vector similarity search and result aggregation logic.
- `trusted/`: The default directory for your personal files to be indexed.
- `data/`: Local storage for training CSVs and audio samples (gitignored).
- `models/`: Local storage for large model weights.
