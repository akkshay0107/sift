import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import src from tests
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
