import pytest
import torch
from faster_whisper import decode_audio

from src.embed.audio import AudioSegment
from src.embed.qwen import QwenEmbedder
from src.embed.whisper_chain import WhisperChain, WhisperTranscriber


@pytest.fixture(scope="module")
def whisper_chain():
    """Fixture to load models once for tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = WhisperTranscriber(model_size="tiny", device=device)
    qwen_embedder = QwenEmbedder()
    return WhisperChain(transcriber, qwen_embedder)


def test_whisper_chain_embedding(whisper_chain, sample_wav):
    """Verifies that Audio -> Whisper -> Qwen pipeline works and produces correct shapes."""
    if not sample_wav.exists():
        pytest.skip(f"Test audio file not found at {sample_wav}")

    audio_data = decode_audio(str(sample_wav), sampling_rate=16000)

    # Create an AudioSegment
    segment = AudioSegment(
        data=audio_data,
        sample_rate=16000,
        t_start=0.0,
        t_end=len(audio_data) / 16000,
        source_id="test_audio",
    )

    # Run embedding
    embedding = whisper_chain.embed(segment)

    # Assertions
    assert isinstance(segment.transcript, str)
    assert len(segment.transcript) > 0
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (2048,)
    assert segment.text_embedding is not None
    assert torch.equal(embedding, segment.text_embedding)


def test_whisper_chain_batch(whisper_chain, sample_wav):
    """Verifies batch processing in WhisperChain."""
    if not sample_wav.exists():
        pytest.skip(f"Test audio file not found at {sample_wav}")

    audio_data = decode_audio(str(sample_wav), sampling_rate=16000)

    # Create two identical segments for testing batching
    segment1 = AudioSegment(
        data=audio_data,
        sample_rate=16000,
        t_start=0.0,
        t_end=len(audio_data) / 16000,
        source_id="test_audio_1",
    )
    segment2 = AudioSegment(
        data=audio_data,
        sample_rate=16000,
        t_start=0.0,
        t_end=len(audio_data) / 16000,
        source_id="test_audio_2",
    )

    segments = [segment1, segment2]
    embeddings = whisper_chain.embed_batch(segments)

    # Assertions
    assert embeddings.shape == (2, 2048)
    assert segment1.transcript is not None
    assert segment2.transcript is not None
    assert segment1.text_embedding is not None
    assert segment2.text_embedding is not None
    assert torch.equal(embeddings[0], segment1.text_embedding)
    assert torch.equal(embeddings[1], segment2.text_embedding)
