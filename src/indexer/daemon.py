import logging
import os
import time
from pathlib import Path

# Suppress tqdm progress bars from Qwen and other third-party libraries.
# tqdm bars write ANSI escape codes to stderr which corrupt plain log output
# when the two streams interleave — subsequent log lines get overwritten or
# swallowed entirely.  The daemon has no interactive UI so progress bars are
# useless here.
os.environ.setdefault("TQDM_DISABLE", "1")

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.indexer.config import MONITORED_DIRECTORIES
from src.indexer.indexer import index_file, index_monitored_directories

logger = logging.getLogger(__name__)


class IndexerEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            self._process_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._process_file(event.src_path)

    def _process_file(self, file_path: str):
        path = Path(file_path)
        try:
            inserted, status = index_file(path)
            if status != "skipped_unchanged":
                logger.info("[Watchdog] Indexed %s -> %d record(s)", path, inserted)
        except Exception as e:
            logger.error("[Watchdog] Failed indexing %s: %s", path, e)


def configure_daemon_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Silence noisy third-party loggers that clutter the daemon output.
    for noisy in ("transformers", "qwen_vl_utils", "easyocr", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def start_indexing_observer(*, perform_initial_scan: bool = True) -> Observer | None:
    if perform_initial_scan:
        logger.info("Performing initial scan of monitored directories...")
        index_monitored_directories()

    observer = Observer()
    event_handler = IndexerEventHandler()

    directories_watched = 0
    for directory in MONITORED_DIRECTORIES:
        if directory.exists():
            observer.schedule(event_handler, str(directory), recursive=True)
            logger.info("Started monitoring: %s", directory)
            directories_watched += 1
        else:
            logger.warning("Monitored directory does not exist: %s", directory)

    if directories_watched == 0:
        logger.error("No valid directories to monitor. Exiting.")
        return None

    observer.start()
    return observer


def run_daemon():
    configure_daemon_logging()
    observer = start_indexing_observer(perform_initial_scan=True)
    if observer is None:
        return

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping daemon...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run_daemon()
