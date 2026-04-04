import json
import os
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config_dir() -> Path:
    system = platform.system()
    if system == "Linux":
        return Path.home() / ".config" / "sift"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "sift"
    elif system == "Windows":
        app_data = os.getenv("APPDATA")
        if app_data:
            return Path(app_data) / "sift"
        return Path.home() / "AppData" / "Roaming" / "sift"
    else:
        return Path.home() / ".sift"


CONFIG_DIR = get_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_or_initialize_config() -> list[Path]:
    if not CONFIG_FILE.exists():
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            default_config = {"monitored_directories": [str(PROJECT_ROOT / "trusted")]}
            with open(CONFIG_FILE, "w") as f:
                json.dump(default_config, f, indent=4)
            print(f"Initialized new config at {CONFIG_FILE}")
        except Exception as e:
            print(f"Warning: Failed to initialize config file at {CONFIG_FILE}: {e}")
        return [PROJECT_ROOT / "trusted"]

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            dirs = config.get("monitored_directories", [str(PROJECT_ROOT / "trusted")])
            return [Path(d) for d in dirs]
    except Exception as e:
        print(f"Error loading config at {CONFIG_FILE}: {e}. Falling back to default.")
        return [PROJECT_ROOT / "trusted"]


MONITORED_DIRECTORIES = load_or_initialize_config()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "catapult_index"

EMBEDDING_DIM = 2048
DISTANCE = "cosine"
