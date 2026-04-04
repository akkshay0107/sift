# Multimodal Memory Engine

This repository hosts a local multimodal embedding engine leveraging huggingface's **Qwen3-VL-Embedding-2B** base model safely stripped of tracking frameworks and completely reliant on PyTorch execution. 

Given an image and/or text string, it passes the input directly into the dense vision/text transformers and exports a unified 2048-dimensional float32 vector, enabling instant multimodal semantic retrieval comparisons.

## 1. Setup Environment
Ensure you have `uv` installed, then add the necessary machine learning dependencies to your local package:

```bash
uv add "transformers>=4.57.0" "qwen-vl-utils>=0.0.14" "torch==2.8.0" pillow torchvision huggingface_hub
```

## 2. Download the Model
You will need to fetch the local copy (~5GB) of the Qwen parameters. Execute this directly using `uv run` to snapshot the payload straight into your local `models` directory:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-VL-Embedding-2B', local_dir='models/Qwen3-VL-Embedding-2B')"
```

## 3. Usage
The Python entry interface (`src/embed/qwen.py`) natively wraps the model routing. 

```python
from src.embed.qwen import QwenEmbedder

# 1. Initialize the class
embedder = QwenEmbedder()

# 2a. Embed Text
text_embedding = embedder.embed("A modern kitchen with wooden cabinets")

# 2b. Embed an Image 
# Automatically maps inputs matching paths/URLs into visual encoding
image_embedding = embedder.embed("tests/test.png")

# 3. Calculate zero-copy L2 Similarity 
score = (text_embedding @ image_embedding.T).item()
print(f"Similarity Score: {score}")
```

### Local Testing
The `tests/` directory uses automated pytest suites to strictly test tensor routing and shape dimensionality.
Run the complete testing matrix:
```bash
uv run pytest tests/
```

## 4. OCR Chaining
You can instantly read raw text from complex images and route it directly into a semantic tensor via the `OCREmbeddingPipeline`.

```python
from src.embed.ocr_chain import OCREmbeddingPipeline

pipeline = OCREmbeddingPipeline()

# Get the concatenated OCR text AND the encoded 2048-dim vector
text, embedding = pipeline.process("tests/document.png", return_embedding=True)

# Or simply extract the text without consuming heavy LLM resources
text = pipeline.process("tests/document.png", return_embedding=False)
```
