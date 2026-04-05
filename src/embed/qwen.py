"""
Qwen3-VL-Embedding-2B wrapper.

Wraps the official Qwen3VLEmbedder shipped with the model to provide a simple
interface: pass a string (text or image path/URL) or a PIL Image and get back
a normalized float32 embedding tensor of shape [1, 2048].

Usage:
    from src.models.embedder import Embedder

    embedder = Embedder()

    text_emb  = embedder.embed("A dog playing on the beach.")
    image_emb = embedder.embed("path/to/photo.jpg")
    pil_emb   = embedder.embed(pil_image_object)

    similarity = (text_emb @ image_emb.T).item()
"""

import sys
import os
from pathlib import Path
from threading import RLock
from typing import Union

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make the model's own scripts/ directory importable so we can use the
# official Qwen3VLEmbedder class without copying it.
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parents[2] / "models" / "Qwen3-VL-Embedding-2B"
_SCRIPTS_DIR = _MODEL_DIR / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402  (lives in model scripts/)

# ---------------------------------------------------------------------------
# Default model path — resolved relative to the project root so it works
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = str(_MODEL_DIR)


class QwenEmbedder:
    """
    Thin wrapper around Qwen3VLEmbedder.

    Parameters
    ----------
    model_path : str
        Path (or HF repo id) for Qwen3-VL-Embedding-2B.
        Defaults to the locally downloaded copy at models/Qwen3-VL-Embedding-2B.
    instruction : str
        System-level instruction prepended to every input.
        Defaults to the model's recommended generic instruction.
    dtype : torch.dtype
        Weight dtype.  bfloat16 is the model's native dtype and runs well on
        Apple Silicon MPS and CUDA.  Use torch.float32 on CPU-only machines.
    **kwargs
        Forwarded to Qwen3VLEmbedder (e.g. max_length, max_pixels …).
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        instruction: str = "",
        dtype: torch.dtype | None = None,
        max_pixels: int = 512
        * 512,  # Limit max pixels to ~260k to prevent CPU exploding on high res
        **kwargs,
    ):
        if dtype is None:
            if torch.cuda.is_available():
                dtype = torch.bfloat16  # CUDA handles bfloat16 natively
            elif torch.backends.mps.is_available():
                dtype = torch.float16  # Best performance for Mac
            else:
                dtype = torch.float32  # CPU fallback

        # Decide if we need to manually move the model to MPS
        move_to_mps = "device_map" not in kwargs and torch.backends.mps.is_available()

        self._instruction = instruction
        self._embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            default_instruction=instruction,
            torch_dtype=dtype,
            max_pixels=max_pixels,
            **kwargs,
        )

        if move_to_mps:
            self._embedder.model.to("mps")
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        input: Union[str, Image.Image],
        instruction: str | None = None,
    ) -> torch.Tensor:
        """
        Embed a single text or image.

        Parameters
        ----------
        input : str | PIL.Image.Image
            • A plain string  →  treated as text.
            • A file path or http/https URL →  treated as an image.
            • A PIL Image object →  treated as an image.
        instruction : str | None
            Override the default instruction for this call only.

        Returns
        -------
        torch.Tensor, shape [1, 2048], dtype float32
            L2-normalized embedding on CPU.
        """
        item = self._build_item(input, instruction)
        with self._lock:
            embedding = self._embedder.process([item])  # shape [1, dim]
        return embedding.float().cpu()

    def embed_batch(
        self,
        inputs: list[Union[str, Image.Image]],
        instruction: str | None = None,
    ) -> torch.Tensor:
        """
        Embed a list of texts and/or images in a single forward pass.

        Parameters
        ----------
        inputs : list of str | PIL.Image.Image
            Mixed list; each element follows the same rules as `embed`.
        instruction : str | None
            Shared instruction override for all items in the batch.

        Returns
        -------
        torch.Tensor, shape [N, 2048], dtype float32
            Row-wise L2-normalized embeddings on CPU.
        """
        items = [self._build_item(inp, instruction) for inp in inputs]
        with self._lock:
            embeddings = self._embedder.process(items)  # shape [N, dim]
        return embeddings.float().cpu()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_item(
        self,
        input: Union[str, Image.Image],
        instruction: str | None,
    ) -> dict:
        """Convert a raw input into the dict format expected by Qwen3VLEmbedder."""
        item: dict = {}

        effective_instruction = (
            instruction if instruction is not None else self._instruction
        )
        if effective_instruction:
            item["instruction"] = effective_instruction

        if isinstance(input, Image.Image):
            item["image"] = input
        elif isinstance(input, str):
            is_url = input.startswith(("http://", "https://", "oss://"))
            is_path = not is_url and os.path.exists(input)

            if is_path:
                suffix = Path(input).suffix.lower()
                video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

                if suffix in video_exts:
                    item["video"] = input
                else:
                    item["image"] = input
            elif is_url:
                lower = input.lower()
                video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

                if lower.endswith(video_exts):
                    item["video"] = input
                else:
                    item["image"] = input
            else:
                item["text"] = input
        else:
            raise TypeError(
                f"embed() expects a str or PIL.Image.Image, got {type(input)}"
            )

        return item
