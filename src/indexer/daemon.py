import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.indexer.config import MONITORED_DIRECTORIES
from src.indexer.indexer import index_file


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
                print(f"[Watchdog] Indexed {path} -> {inserted} record(s)")
        except Exception as e:
            print(f"[Watchdog] Failed indexing {path}: {e}")


def run_daemon():
    observer = Observer()
    event_handler = IndexerEventHandler()

    directories_watched = 0
    for directory in MONITORED_DIRECTORIES:
        if directory.exists():
            observer.schedule(event_handler, str(directory), recursive=True)
            print(f"Started monitoring: {directory}")
            directories_watched += 1
        else:
            print(f"Warning: Monitored directory does not exist: {directory}")

    if directories_watched == 0:
        print("No valid directories to monitor. Exiting.")
        return

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping daemon...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run_daemon()
