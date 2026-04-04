import pytest
import torch

from src.embed.ocr_chain import OCREmbeddingPipeline


@pytest.fixture(scope="module")
def pipeline():
    return OCREmbeddingPipeline()


def test_ocr_pipeline(pipeline, test_png):
    """Verifies that the entire Image -> EasyOCR -> String -> Tensor pipeline connects properly."""
    if not test_png.exists():
        pytest.skip("Test image missing.")

    # Test 1: Full pipeline
    text, embedding = pipeline.process(str(test_png), return_embedding=True)

    assert isinstance(text, str)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, 2048)


def test_ocr_only(pipeline, test_png):
    """Verifies that the user can skip the 2B param embedding bottleneck and just get text."""
    if not test_png.exists():
        pytest.skip("Test image missing.")

    text = pipeline.process(str(test_png), return_embedding=False)

    assert isinstance(text, str)
    assert type(text) is not tuple  # Ensures only string was returned
