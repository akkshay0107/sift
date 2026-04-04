import easyocr

class OCREngine:
    """
    Wrapper around EasyOCR to process images and extract text.
    """
    def __init__(self, languages=['en']):
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
