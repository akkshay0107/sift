from typing import Optional, Tuple, Union

import easyocr
import torch

from src.embed.qwen import QwenEmbedder


class OCREngine:
    """
    Wrapper around EasyOCR to process images and extract text.
    """

    def __init__(self, languages=["en"]):
        # Uses GPU if available, else falls back to CPU automatically
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image_path: str) -> str:
        """
        Extracts all visible text from the given image path and
        returns it as a single concatenated string.
        """
        # Readtext returns a list of tuples: (bounding_box, string, confidence)
        result = self.reader.readtext(image_path)

        # We only need the text portion (index 1) for embedding
        text_blocks = [item[1] for item in result]

        # Consolidate into a single contiguous string
        return " ".join(text_blocks)


class OCREmbeddingPipeline:
    """
    Orchestrates the pipeline: Image -> OCR Engine -> Text String -> Qwen3-VL-Embedding -> Vector.
    """

    def __init__(
        self,
        embedder: Optional[QwenEmbedder] = None,
        embedder_kwargs: Optional[dict] = None,
    ):
        self.ocr = OCREngine()

        if embedder is not None:
            # Reuse an already-loaded QwenEmbedder (preferred — avoids loading a
            # second copy of the 5 GB model when the caller already has one).
            self.embedder = embedder
        else:
            # Fallback: create a new instance (e.g. when used standalone).
            embedder_kwargs = embedder_kwargs or {}
            self.embedder = QwenEmbedder(**embedder_kwargs)

    def process(
        self,
        image_path: str,
        return_embedding: bool = True,
        instruction: Optional[str] = None,
    ) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        Executes the pipeline over an image.

        Args:
            image_path: Path to the local image.
            return_embedding: If True, calls the embedding model. If False, skips model entirely.
            instruction: Optional system instruction for the embedding model.

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
        tensor = self.embedder.embed(text, instruction=instruction)

        return text, tensor
