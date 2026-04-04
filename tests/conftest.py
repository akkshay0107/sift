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
    return monitored_dir / "sample.wav"


@pytest.fixture(scope="session")
def test_png(monitored_dir):
    return monitored_dir / "test.png"
