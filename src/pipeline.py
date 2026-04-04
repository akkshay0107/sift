from typing import Optional, Tuple, Union
import torch
from src.extract.ocr import OCREngine
from src.embed.qwen import QwenEmbedder

class OCREmbeddingPipeline:
    """
    Orchestrates the pipeline: Image -> OCR Engine -> Text String -> Qwen3-VL-Embedding -> Vector.
    """
    def __init__(self, embedder_kwargs: Optional[dict] = None):
        self.ocr = OCREngine()
        
        # Initialize embedder with any custom configurations required (e.g. max_pixels)
        embedder_kwargs = embedder_kwargs or {}
        self.embedder = QwenEmbedder(**embedder_kwargs)

    def process(self, image_path: str, return_embedding: bool = True) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        Executes the pipeline over an image.
        
        Args:
            image_path: Path to the local image.
            return_embedding: If True, calls the embedding model. If False, skips model entirely.
            
        Returns:
            If return_embedding is True: (extracted_text, embedding_tensor)
            If return_embedding is False: extracted_text
        """
        # Step 1: Run EasyOCR to fetch raw text
        text = self.ocr.extract_text(image_path)
        
        # Step 2: Skip embedding if user strictly only wanted text
        if not return_embedding:
            return text
            
        # Step 3: Embed the extracted text using Qwen
        # If the image was empty or had no text, we can still embed empty string or skip
        tensor = self.embedder.embed(text)
        
        return text, tensor
