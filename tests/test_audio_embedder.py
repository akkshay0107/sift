import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.embed.audio import AudioEmbedder


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sample_rate = f.getframerate()
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    samples = np.frombuffer(raw, dtype=dtype_map.get(sampwidth, np.int16)).astype(
        np.float32
    )
    samples /= float(2 ** (8 * sampwidth - 1))

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sample_rate


@pytest.fixture(scope="module")
def audio_embedder():
    return AudioEmbedder()


@pytest.fixture(scope="module")
def sample_audio(sample_wav):
    if not sample_wav.exists():
        pytest.skip(f"sample.wav not found at {sample_wav}")
    return _load_wav(sample_wav)


def test_audio_embedding_shape(audio_embedder, sample_audio):
    waveform, sample_rate = sample_audio
    emb = audio_embedder.embed(waveform, sample_rate)

    print(f"\nEmbedding: {emb}")
    print(f"Shape:     {emb.shape}")
    print(f"Dtype:     {emb.dtype}")
    print(f"Norm:      {emb.norm().item():.6f}")

    assert emb.shape == (1, 2048)
    assert not torch.isnan(emb).any().item(), "Audio embedding contains NaNs"
    assert not torch.isinf(emb).any().item(), "Audio embedding contains Infs"


def test_audio_embedding_dtype(audio_embedder, sample_audio):
    waveform, sample_rate = sample_audio
    emb = audio_embedder.embed(waveform, sample_rate)
    assert emb.dtype == torch.float32


def test_audio_embedding_normalized(audio_embedder, sample_audio):
    waveform, sample_rate = sample_audio
    emb = audio_embedder.embed(waveform, sample_rate)
    assert abs(emb.norm().item() - 1.0) < 1e-4


def test_audio_embedding_on_cpu(audio_embedder, sample_audio):
    waveform, sample_rate = sample_audio
    emb = audio_embedder.embed(waveform, sample_rate)
    assert emb.device.type == "cpu"


def test_embed_batch_shape(audio_embedder, sample_audio):
    waveform, sample_rate = sample_audio
    embs = audio_embedder.embed_batch(
        [(waveform, sample_rate), (waveform, sample_rate)]
    )

    print(f"\nBatch shape: {embs.shape}")

    assert embs.shape == (2, 2048)
    assert embs.dtype == torch.float32
