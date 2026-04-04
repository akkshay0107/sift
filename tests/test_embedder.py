import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    # 2. Assert Semantic Validity (L2 normalized similarity)
    sim_similar = (emb1 @ emb2.T).item()
    sim_different = (emb1 @ emb3.T).item()
    
    assert sim_similar > sim_different
    assert sim_similar > 0.4
    assert sim_different < 0.2

def test_image_embedding_shape(embedder):
    """Verifies that an image input correctly routes to the vision processor without exploding."""
    test_dir = Path(__file__).parent
    image_path = test_dir / "test.png"
    
    # Skip if the user deleted the image
    if not image_path.exists():
        pytest.skip(f"Test image not found at {image_path}")
        
    emb_img = embedder.embed(str(image_path))
    
    assert isinstance(emb_img, torch.Tensor)
    assert emb_img.shape == (1, 2048)
