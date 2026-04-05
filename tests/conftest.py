import sys
from pathlib import Path

import pytest

# Add project root to sys.path so we can import src from tests
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def monitored_dir():
    return project_root / "trusted"


@pytest.fixture(scope="session")
def sample_wav(monitored_dir):
    return monitored_dir / "speech_00.wav"


@pytest.fixture(scope="session")
def test_png(monitored_dir):
    return monitored_dir / "flickr8k_00.jpg"


@pytest.fixture(scope="session")
def test_video(monitored_dir):
    return monitored_dir / "jester.mp4"


@pytest.fixture(scope="session")
def test_txt(monitored_dir):
    return monitored_dir / "passage_00.txt"


@pytest.fixture(scope="session")
def test_noise(monitored_dir):
    return monitored_dir / "noise_00_crow.wav"
