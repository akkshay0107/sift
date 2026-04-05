import pytest
import torch

from src.embed.qwen import QwenEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Fixture to load the model once for all tests."""
    return QwenEmbedder()


def test_text_embedding_shape_and_similarity(embedder):
    """Verifies that text inputs produce the correct tensor shape and semantic correlations."""
    text1 = "A woman playing with her dog on a beach at sunset."
    text2 = "A happy golden retriever running on the sand."
    text3 = "The stock market went up today by 5%."

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    emb3 = embedder.embed(text3)

    # 1. Assert Output Types and Shapes
    assert isinstance(emb1, torch.Tensor)
    assert emb1.shape == (1, 2048)

    # 1b. Debug/Stability Checks (Ensure float16 didn't overflow to NaN/Inf)
    assert not torch.isnan(emb1).any().item(), (
        "Qwen text embedding contains NaNs (likely float16 overflow)"
    )
    assert not torch.isinf(emb1).any().item(), "Qwen text embedding contains Infs"
    assert not torch.isnan(emb2).any().item(), "Qwen text embedding contains NaNs"
    assert not torch.isnan(emb3).any().item(), "Qwen text embedding contains NaNs"

    # 2. Assert Semantic Validity (L2 normalized similarity)
    sim_similar = (emb1 @ emb2.T).item()
    sim_different = (emb1 @ emb3.T).item()

    assert sim_similar > sim_different
    assert sim_similar > 0.4
    assert sim_different < 0.2


def test_embedding_from_file(embedder, test_txt):
    """Verifies that reading text from a file and embedding it works."""
    if not test_txt.exists():
        pytest.skip(f"Test text file not found at {test_txt}")

    with open(test_txt, "r") as f:
        text = f.read()

    emb = embedder.embed(text[:1000])  # Use a snippet
    assert emb.shape == (1, 2048)
    assert not torch.isnan(emb).any().item()


def test_image_embedding_shape(embedder, test_png):
    """Verifies that an image input correctly routes to the vision processor without exploding."""
    if not test_png.exists():
        pytest.skip(f"Test image not found at {test_png}")

    emb_img = embedder.embed(str(test_png))

    assert isinstance(emb_img, torch.Tensor)
    assert emb_img.shape == (1, 2048)

    # 3. Debug/Stability Checks
    assert not torch.isnan(emb_img).any().item(), "Qwen image embedding contains NaNs"
    assert not torch.isinf(emb_img).any().item(), "Qwen image embedding contains Infs"
