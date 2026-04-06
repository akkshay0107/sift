from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def get_pipelines_for_file(path: Path) -> list[str]:
    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return ["text"]

    if ext in IMAGE_EXTENSIONS:
        return ["image", "ocr_text"]

    if ext in AUDIO_EXTENSIONS:
        return ["audio", "transcript_text"]

    if ext in VIDEO_EXTENSIONS:
        return ["video"]

    return []
