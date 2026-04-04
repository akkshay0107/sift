import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from src.embed.ocr_chain import OCREmbeddingPipeline

@pytest.fixture(scope="module")
def pipeline():
    return OCREmbeddingPipeline()

def test_ocr_pipeline(pipeline):
    """Verifies that the entire Image -> EasyOCR -> String -> Tensor pipeline connects properly."""
    test_dir = Path(__file__).parent
    image_path = test_dir / "test.png"
    
    if not image_path.exists():
        pytest.skip("Test image missing.")
        
    # Test 1: Full pipeline
    text, embedding = pipeline.process(str(image_path), return_embedding=True)
    
    assert isinstance(text, str)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, 2048)
    
def test_ocr_only(pipeline):
    """Verifies that the user can skip the 2B param embedding bottleneck and just get text."""
    test_dir = Path(__file__).parent
    image_path = test_dir / "test.png"
    
    if not image_path.exists():
        pytest.skip("Test image missing.")
        
    text = pipeline.process(str(image_path), return_embedding=False)
    
    assert isinstance(text, str)
    assert type(text) is not tuple # Ensures only string was returned
