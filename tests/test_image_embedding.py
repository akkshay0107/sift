import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.embed.qwen import QwenEmbedder

def main():
    print("Initializing Qwen3-VL-Embedding-2B...")
    embedder = QwenEmbedder()
    
    # Text to test against
    query_text = "A modern kitchen with wooden cabinets"
    print(f"\nQuery Text: '{query_text}'")
    text_emb = embedder.embed(query_text)
    
    # We will pick the first image found in the 'tests' directory for this test
    # (or you can specify a direct path)
    test_dir = Path(__file__).parent
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    found_images = [f for f in os.listdir(test_dir) if Path(f).suffix.lower() in image_extensions]
    
    if not found_images:
        print("\n[!] No images found in the 'tests' directory. Please drop an image in here and re-run!")
        sys.exit(1)
        
    image_path = test_dir / found_images[0]
    print(f"\nFound Image: {image_path}")
    
    # Process image embedding
    print("Processing image embedding...")
    image_emb = embedder.embed(str(image_path))
    print(f"Image Shape: {image_emb.shape}")
    
    # Calculate similarity
    sim_score = (text_emb @ image_emb.T).item()
    print("\n--- Similarity Score ---")
    print(f"Similarity (Text vs Image): {sim_score:.4f}")
    
if __name__ == '__main__':
    main()
