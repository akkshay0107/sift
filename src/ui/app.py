from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.search import SearchResult, search_similar, search_similar_files


ACCENT_PINK = "#ff1493"
ACCENT_TAN = "#b8968f"
PANEL_BG = "#171515"
SURFACE_BG = "#0b0b0b"
TEXT_PRIMARY = "#f3eee8"
TEXT_MUTED = "#9d847c"
GRID_LINE = "#2c2a2a"


@dataclass(slots=True)
class EntityRow:
    label: str
    location: str


class GridBackgroundWidget(QWidget):
    pass


class MemoryPanel(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("memoryPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        layout.addWidget(self.title_label)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("panelList")
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setSpacing(6)
        layout.addWidget(self.list_widget, 1)

        self.footer_label = QLabel("")
        self.footer_label.setObjectName("panelFooter")
        layout.addWidget(self.footer_label)

    def set_items(self, items: list[str], *, highlight_index: int = 0) -> None:
        self.list_widget.clear()
        for index, value in enumerate(items):
            item = QListWidgetItem(value)
            if index == highlight_index:
                item.setForeground(QColor(ACCENT_PINK))
            else:
                item.setForeground(QColor(TEXT_PRIMARY if index == 0 else TEXT_MUTED))
            self.list_widget.addItem(item)


class EntityList(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("entityPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("entityHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)

        self.title_label = QLabel("RECOGNIZED_ENTITIES")
        self.title_label.setObjectName("entityHeaderTitle")
        self.filter_label = QLabel("FILTER: ALL")
        self.filter_label.setObjectName("entityHeaderFilter")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.filter_label)
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("entityList")
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setSpacing(0)
        layout.addWidget(self.list_widget, 1)

    def set_rows(self, rows: list[EntityRow]) -> None:
        self.list_widget.clear()
        for row in rows:
            item = QListWidgetItem(f"{row.label}    LOC:{row.location}")
            item.setForeground(QColor(TEXT_PRIMARY))
            self.list_widget.addItem(item)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Query Memory")
        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1360, 900)
        self.setMinimumSize(1100, 760)
        self._results: list[SearchResult] = []
        self._file_paths: list[str] = []

        self._build_ui()
        self._apply_styles()

        close_shortcut = QShortcut(QKeySequence("Esc"), self)
        close_shortcut.activated.connect(self.close)

    def _build_ui(self) -> None:
        root = GridBackgroundWidget()
        self.setCentralWidget(root)

        page_layout = QVBoxLayout(root)
        page_layout.setContentsMargins(160, 18, 160, 40)
        page_layout.setSpacing(10)

        self.header = self._build_header()
        page_layout.addWidget(self.header, 0, Qt.AlignTop)

        self.results_shell = QFrame()
        self.results_shell.setObjectName("shell")
        self.results_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_shell.hide()
        page_layout.addWidget(self.results_shell, 1)

        shell_layout = QVBoxLayout(self.results_shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        shell_layout.addWidget(self._build_top_panels())
        self.entity_list = EntityList()
        shell_layout.addWidget(self.entity_list, 1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(30, 28, 30, 28)
        layout.setSpacing(18)

        icon = QLabel("⌕")
        icon.setObjectName("searchIcon")
        layout.addWidget(icon)

        self.query_input = QLineEdit()
        self.query_input.setObjectName("queryInput")
        self.query_input.setPlaceholderText("QUERY MEMORY")
        self.query_input.returnPressed.connect(self.run_search)
        layout.addWidget(self.query_input, 1)

        self.cancel_label = QLabel("ESC_TO_CANCEL")
        self.cancel_label.setObjectName("cancelLabel")
        layout.addWidget(self.cancel_label)

        return header

    def _build_top_panels(self) -> QWidget:
        wrapper = QFrame()
        wrapper.setObjectName("topPanels")
        layout = QGridLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.files_panel = MemoryPanel("FILES")
        self.matches_panel = MemoryPanel("TOP_MATCHES")
        self.metadata_panel = MemoryPanel("METADATA")

        self.search_button = QPushButton("RUN QUERY")
        self.search_button.setObjectName("queryButton")
        self.search_button.clicked.connect(self.run_search)
        self.matches_panel.layout().addWidget(self.search_button)

        layout.addWidget(self.files_panel, 0, 0)
        layout.addWidget(self.matches_panel, 0, 1)
        layout.addWidget(self.metadata_panel, 0, 2)
        return wrapper

    def _apply_styles(self) -> None:
        font = QFont("Bahnschrift")
        font.setStyleHint(QFont.SansSerif)
        QApplication.instance().setFont(font)

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: transparent;
            }}
            #shell {{
                background: {SURFACE_BG};
                border: 4px solid #e7e2da;
            }}
            #header {{
                background: #0f0f0f;
                border: 4px solid #e7e2da;
            }}
            #searchIcon {{
                color: {ACCENT_PINK};
                font-size: 34px;
                font-weight: 700;
            }}
            #queryInput {{
                background: transparent;
                border: none;
                border-right: 4px solid {ACCENT_PINK};
                color: {TEXT_PRIMARY};
                font-size: 28px;
                font-weight: 700;
                padding: 8px 12px 8px 0;
            }}
            #queryInput::placeholder {{
                color: {TEXT_PRIMARY};
            }}
            #cancelLabel {{
                color: #5d4d49;
                font-size: 16px;
                letter-spacing: 2px;
            }}
            #topPanels {{
                border-bottom: 3px solid #e7e2da;
            }}
            #memoryPanel, #entityPanel {{
                background: {PANEL_BG};
                border-right: 1px solid #d7d0c8;
            }}
            #memoryPanel:last-child {{
                border-right: none;
            }}
            #panelTitle {{
                color: {TEXT_PRIMARY};
                font-size: 18px;
                font-weight: 700;
                letter-spacing: 2px;
            }}
            #panelList {{
                background: transparent;
                color: {TEXT_PRIMARY};
                font-size: 16px;
                outline: none;
            }}
            #panelList::item {{
                padding: 8px 6px;
                border: none;
            }}
            #panelList::item:selected {{
                background: transparent;
                border: none;
            }}
            #panelFooter {{
                color: {ACCENT_PINK};
                font-size: 15px;
                font-weight: 700;
                border-top: 1px solid #3a3434;
                padding-top: 12px;
            }}
            #queryButton {{
                background: transparent;
                border: 3px solid {ACCENT_PINK};
                color: {ACCENT_PINK};
                font-size: 15px;
                font-weight: 700;
                min-height: 44px;
            }}
            #queryButton:hover {{
                background: rgba(255, 20, 147, 0.10);
            }}
            #entityHeader {{
                background: #0f0f0f;
                border-bottom: 1px solid #d7d0c8;
            }}
            #entityHeaderTitle {{
                color: {ACCENT_PINK};
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 3px;
            }}
            #entityHeaderFilter {{
                color: {TEXT_MUTED};
                font-size: 15px;
            }}
            #entityList {{
                background: #101010;
                color: {TEXT_PRIMARY};
                font-size: 16px;
                outline: none;
            }}
            #entityList::item {{
                padding: 20px 18px;
                border-bottom: 1px solid #2f2929;
            }}
            """
        )

    def run_search(self) -> None:
        prompt = self.query_input.text().strip()
        if not prompt:
            return

        try:
            file_matches = search_similar_files(prompt, k=8, score_threshold=0.2)
            raw_matches = search_similar(prompt, k=8, score_threshold=0.2)
        except Exception as exc:
            QMessageBox.critical(self, "Search failed", str(exc))
            return

        self._results = raw_matches
        self._file_paths = [match.source_path for match in file_matches]
        self._update_from_search(file_matches, raw_matches)
        self.results_shell.show()

    def _update_from_search(
        self,
        file_matches: list[Any],
        raw_matches: list[SearchResult],
    ) -> None:
        file_labels = [match.file_name for match in file_matches] or ["NO_MATCHING_FILES"]
        self.files_panel.set_items(file_labels)
        self.files_panel.footer_label.setText(f"{len(file_matches)} FILES LOADED")

        top_match_labels = [
            f"{match.file_name}  [{match.score:.3f}]"
            for match in file_matches[:3]
        ]
        self.matches_panel.set_items(top_match_labels or ["NO_ACTIVE_MATCHES"])
        self.matches_panel.footer_label.setText("RESULTS REFRESHED")

        metadata_lines = self._build_metadata_lines(raw_matches)
        self.metadata_panel.set_items(metadata_lines or ["NO_METADATA"])
        self.metadata_panel.footer_label.setText("LIVE PAYLOAD SNAPSHOT")

        entity_rows = self._build_entity_rows(raw_matches)
        self.entity_list.set_rows(entity_rows or [EntityRow("NO_RECOGNIZED_ENTITIES", "/")])

    def _build_metadata_lines(self, raw_matches: list[SearchResult]) -> list[str]:
        lines: list[str] = []
        for result in raw_matches[:3]:
            payload = result.payload or {}
            extension = str(payload.get("extension") or "n/a").upper()
            modality = str(payload.get("modality") or "unknown").upper()
            pipeline = str(payload.get("pipeline_name") or "pipeline").upper()
            lines.append(f"{extension} / {modality} / {pipeline}")
        return lines

    def _build_entity_rows(self, raw_matches: list[SearchResult]) -> list[EntityRow]:
        rows: list[EntityRow] = []
        seen: set[tuple[str, str]] = set()
        for result in raw_matches:
            payload = result.payload or {}
            file_name = str(payload.get("file_name") or "UNKNOWN")
            source_path = Path(str(payload.get("source_path") or "/"))
            label = file_name.upper()
            location = str(source_path.parent).replace("\\", "/").upper()
            key = (label, location)
            if key in seen:
                continue
            seen.add(key)
            rows.append(EntityRow(label, location))
            if len(rows) == 6:
                break
        return rows


def launch_desktop_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.showFullScreen()
    return app.exec()
