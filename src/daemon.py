from __future__ import annotations

import faulthandler
import logging
import sys

from src.indexer.daemon import configure_daemon_logging, start_indexing_observer
from src.indexer.pipelines import preload_shared_models
from src.ui import launch_desktop_app

logger = logging.getLogger(__name__)


def run_main_daemon() -> int:
    faulthandler.enable(all_threads=True, file=sys.stderr)
    configure_daemon_logging()
    logger.info("Preloading shared models...")
    try:
        preload_shared_models(qwen=True)
    except Exception:
        logger.exception("Shared model preload failed.")
        return 1
    logger.info("Shared model preload complete.")

    observer = start_indexing_observer(perform_initial_scan=True)
    if observer is None:
        return 1

    logger.info("Starting unified daemon UI in hidden mode...")
    try:
        return launch_desktop_app(show_on_start=True)
    finally:
        logger.info("Stopping filesystem observer...")
        observer.stop()
        observer.join()


if __name__ == "__main__":
    raise SystemExit(run_main_daemon())
