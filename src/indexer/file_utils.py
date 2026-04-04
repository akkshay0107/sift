import hashlib
import mimetypes
from pathlib import Path


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def guess_mime_type(path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type


def file_extension(path: Path) -> str:
    return path.suffix.lower()
