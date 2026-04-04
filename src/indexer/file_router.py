from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_pipelines_for_file(path: Path) -> list[str]:
    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return ["text"]

    if ext in IMAGE_EXTENSIONS:
        return ["image", "ocr_text"]

    return []



# need to add functionality for others also