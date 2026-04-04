import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.embed.qwen import QwenEmbedder

def main():
    print("Initializing Qwen3-VL-Embedding-2B...")
    # This will load the model from your local 'models/Qwen3-VL-Embedding-2B' directory
    embedder = QwenEmbedder()
    
    print("\nModel initialized! Testing text embeddings...")
    
    # Test 1: Single text
    text1 = "A woman playing with her dog on a beach at sunset."
    print(f"\nEmbedding 1: '{text1}'")
    emb1 = embedder.embed(text1)
    print(f"Shape: {emb1.shape}")
    
    # Test 2: Another text
    text2 = "A happy golden retriever running on the sand."
    print(f"\nEmbedding 2: '{text2}'")
    emb2 = embedder.embed(text2)
    print(f"Shape: {emb2.shape}")

    # Test 3: Unrelated text
    text3 = "The stock market went up today by 5%."
    print(f"\nEmbedding 3: '{text3}'")
    emb3 = embedder.embed(text3)
    
    # Calculate similarities (dot product since they are already L2 normalized)
    print("\n--- Similarity Scores ---")
    
    # We use .item() to pull the single float value out of the PyTorch tensor
    sim_1_2 = (emb1 @ emb2.T).item()
    sim_1_3 = (emb1 @ emb3.T).item()

    print(f"Similarity ('{text1}' vs '{text2}'): {sim_1_2:.4f} (Should be high)")
    print(f"Similarity ('{text1}' vs '{text3}'): {sim_1_3:.4f} (Should be lower)")
    
if __name__ == '__main__':
    main()
