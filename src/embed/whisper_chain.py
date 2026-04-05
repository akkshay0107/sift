from typing import List

import numpy as np
import torch

try:
    from faster_whisper import WhisperModel

    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

from src.embed.audio import AudioSegment
from src.embed.qwen import QwenEmbedder


class WhisperTranscriber:
    """
    Wrapper for faster-whisper.
    Converts audio segments into text transcripts.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        if not _WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            )

        self.model_size = model_size
        self.device = device
        # Use float16 on GPU, int8 on CPU for performance
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Runs Whisper on a single audio buffer.
        Returns the combined transcript text.

        Args:
            audio: 1-D float32 numpy array (16kHz).
        """
        segments, _ = self.model.transcribe(audio, beam_size=5)
        return " ".join([s.text for s in segments]).strip()


class WhisperChain:
    """
    Orchestrates the Audio -> Text -> Embedding pipeline.
    Matches the API of AudioEmbedder and QwenEmbedder.

    Args:
        transcriber: Configured WhisperTranscriber.
        qwen_embedder: Configured QwenEmbedder.
    """

    def __init__(self, transcriber: WhisperTranscriber, qwen_embedder: QwenEmbedder):
        self.transcriber = transcriber
        self.qwen_embedder = qwen_embedder

    @torch.no_grad()
    def embed(self, segment: AudioSegment) -> torch.Tensor:
        """
        Produce a single 2048-dim L2-normalised text embedding for *segment*
        by first transcribing its audio and then embedding the text.

        Populates segment.transcript and segment.text_embedding.

        Args:
            segment: ``AudioSegment`` with populated ``data``.

        Returns:
            1-D ``torch.Tensor`` of shape ``(2048,)`` on CPU.
        """
        # 1. Audio -> Text
        segment.transcript = self.transcriber.transcribe(segment.data)

        # 2. Text -> Embedding
        # If transcription is empty, we still embed the empty string to get a vector
        # in the Qwen space (or handle as preferred).
        text = segment.transcript if segment.transcript else ""
        emb = self.qwen_embedder.embed(text, instruction="").squeeze(0)  # (2048,)
        segment.text_embedding = emb

        return emb

    @torch.no_grad()
    def embed_batch(self, segments: List[AudioSegment]) -> torch.Tensor:
        """
        Produce text embeddings for a list of segments in a single pass.
        Populates transcript and text_embedding for each segment.

        Args:
            segments: List of ``AudioSegment`` objects.

        Returns:
            Tensor of shape ``(N, 2048)`` on CPU, one row per segment.
        """
        # 1. Sequential Transcription (Whisper doesn't batch easily across buffers)
        for segment in segments:
            segment.transcript = self.transcriber.transcribe(segment.data)

        # 2. Batch Text Embedding
        texts = [s.transcript if s.transcript else "" for s in segments]
        embeddings = self.qwen_embedder.embed_batch(texts)  # type: ignore

        for i, segment in enumerate(segments):
            segment.text_embedding = embeddings[i]

        return embeddings
