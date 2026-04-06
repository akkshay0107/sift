from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pynput import keyboard
from PySide6.QtCore import (
    QEasingCurve,
    QItemSelectionModel,
    QObject,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.search import SearchResult, search_similar
from src.search.bundler import SearchBundle, build_bundles

# Dev-facing theme knob.
ACCENT_COLOR = "#5a8fd6"
BOUNDARY_COLOR = "rgba(90, 143, 214, 0.55)"
BOUNDARY_COLOR_STRONG = "rgba(90, 143, 214, 0.75)"
ACCENT_TAN = "#b8968f"
PANEL_BG = "#171515"
SURFACE_BG = "#0b0b0b"
TEXT_PRIMARY = "#f3eee8"
TEXT_MUTED = "#8792a1"
GRID_LINE = "#2c2a2a"
ACCENT_SOFT = "rgba(90, 143, 214, 0.18)"
ACCENT_FAINT = "rgba(90, 143, 214, 0.10)"
UI_FONT_FAMILY = "Funnel Display"
UI_FONT_FILE: str | None = str(
    Path(__file__).parent.parent.parent
    / "assets"
    / "fonts"
    / "FunnelDisplay-VariableFont_wght.ttf"
)
UI_FONT_FALLBACKS = []


class HotkeySignaler(QObject):
    triggered = Signal()


@dataclass(slots=True)
class EntityRow:
    label: str
    location: str
    source_path: str


class GridBackgroundWidget(QWidget):
    pass


class MemoryPanel(QFrame):
    def __init__(
        self,
        title: str,
        *,
        open_handler: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("memoryPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._open_handler = open_handler

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        layout.addWidget(self.title_label)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("panelList")
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setSpacing(6)
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setFocusPolicy(Qt.StrongFocus)
        self.list_widget.setCursor(Qt.PointingHandCursor)
        self.list_widget.viewport().setCursor(Qt.PointingHandCursor)
        self.list_widget.viewport().setAutoFillBackground(False)
        self.list_widget.itemDoubleClicked.connect(self._handle_item_clicked)
        self.list_widget.itemActivated.connect(self._handle_item_clicked)
        layout.addWidget(self.list_widget, 1)

        self.footer_label = QLabel("")
        self.footer_label.setObjectName("panelFooter")
        layout.addWidget(self.footer_label)

    def set_items(self, items: list[str], *, highlight_index: int = 0) -> None:
        self.list_widget.clear()
        for value in items:
            item = QListWidgetItem(value)
            item.setForeground(QColor(TEXT_PRIMARY))
            self.list_widget.addItem(item)

    def set_bundle(self, bundle: SearchBundle | None) -> None:
        if bundle is None:
            self.title_label.setText("No bundle")
            self.set_items(["No matching bundle found"])
            self.footer_label.setText("0 files")
            return

        self.list_widget.clear()
        seen: set[str] = set()
        for source_path in bundle.source_files:
            file_name = Path(source_path).name if source_path else ""
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, source_path)
            item.setForeground(QColor(TEXT_PRIMARY))
            self.list_widget.addItem(item)

        if self.list_widget.count() == 0:
            for view in bundle.views:
                payload = view.payload or {}
                file_name = str(payload.get("file_name") or "").strip()
                source_path = str(payload.get("source_path") or "").strip()
                if not file_name or file_name in seen:
                    continue
                seen.add(file_name)
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, source_path)
                item.setForeground(QColor(TEXT_PRIMARY))
                self.list_widget.addItem(item)

        if self.list_widget.count() == 0:
            self.set_items(["NO_FILES"])
        elif self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0, QItemSelectionModel.ClearAndSelect)

        file_count = len(seen)
        self.footer_label.setText(f"{file_count} file{'s' if file_count != 1 else ''}")

    def _handle_item_clicked(self, item: QListWidgetItem) -> None:
        if self._open_handler is None:
            return
        source_path = item.data(Qt.UserRole)
        if isinstance(source_path, str) and source_path:
            self._open_handler(source_path)

    def _format_bundle_title(self, bundle: SearchBundle) -> str:
        title = bundle.title.strip() or "BUNDLE"
        path = Path(title)
        display = path.stem if path.suffix else title
        return display.replace("_", " ").upper()


class EntityList(QFrame):
    def __init__(
        self,
        *,
        open_handler: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("entityPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._open_handler = open_handler

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("entityHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)

        self.title_label = QLabel("Source Files")
        self.title_label.setObjectName("entityHeaderTitle")
        self.filter_label = QLabel("All results")
        self.filter_label.setObjectName("entityHeaderFilter")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.filter_label)
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("entityList")
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setSpacing(0)
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setFocusPolicy(Qt.StrongFocus)
        self.list_widget.setCursor(Qt.PointingHandCursor)
        self.list_widget.viewport().setCursor(Qt.PointingHandCursor)
        self.list_widget.viewport().setAutoFillBackground(False)
        self.list_widget.itemDoubleClicked.connect(self._handle_item_clicked)
        self.list_widget.itemActivated.connect(self._handle_item_clicked)
        layout.addWidget(self.list_widget, 1)

        # Footer bar — acts as a visual buffer between scrollable list and
        # the rounded bottom corners, identical to the MemoryPanel footer.
        self.footer_bar = QFrame()
        self.footer_bar.setObjectName("entityFooter")
        self.footer_bar.setFixedHeight(24)
        layout.addWidget(self.footer_bar)

    def set_rows(self, rows: list[EntityRow]) -> None:
        self.list_widget.clear()
        for row in rows:
            item = QListWidgetItem(f"{row.label}    LOC:{row.location}")
            item.setForeground(QColor(TEXT_PRIMARY))
            item.setData(Qt.UserRole, row.source_path)
            self.list_widget.addItem(item)
        self.list_widget.clearSelection()

    def _handle_item_clicked(self, item: QListWidgetItem) -> None:
        if self._open_handler is None:
            return
        source_path = item.data(Qt.UserRole)
        if isinstance(source_path, str) and source_path:
            self._open_handler(source_path)


class MainWindow(QMainWindow):
    def __init__(self, *, show_on_start: bool = True) -> None:
        super().__init__()
        self._show_on_start = show_on_start
        self.setWindowTitle("Query Memory")
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1020, 900)
        self.setMinimumSize(820, 760)

        # Center horizontally native to PySide
        if QApplication.primaryScreen():
            screen_geo = QApplication.primaryScreen().availableGeometry()
            x = screen_geo.x() + (screen_geo.width() - self.width()) // 2
            self.move(x, screen_geo.top())

        self._results: list[SearchResult] = []
        self._bundle_panels: list[MemoryPanel] = []
        self._last_bundle_focus_index = 0
        self._results_visible = False
        self._active_animations: list[QParallelAnimationGroup] = []

        self._build_ui()
        self._apply_styles()

        close_shortcut = QShortcut(QKeySequence("Esc"), self)
        close_shortcut.activated.connect(self._hide_and_reset)

        if show_on_start:
            QTimer.singleShot(0, self.play_intro_animation)
        self._start_global_hotkey()

    def _hide_and_reset(self) -> None:
        self.hide()

    def _start_global_hotkey(self) -> None:
        self._hotkey_signaler = HotkeySignaler()
        self._hotkey_signaler.triggered.connect(self._toggle_visibility)

        def on_activate():
            self._hotkey_signaler.triggered.emit()

        self._listener = keyboard.GlobalHotKeys({"<ctrl>+<space>": on_activate})
        self._listener.start()

    def _toggle_visibility(self) -> None:
        if self.isVisible():
            self._hide_and_reset()
        else:
            self.show()
            self.play_intro_animation()
            if self._results_visible:
                self.play_results_animation()
            self._apply_macos_overlay()
            self.query_input.setFocus()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(80, self._apply_macos_overlay)

    def _apply_macos_overlay(self) -> None:
        if platform.system() != "Darwin":
            return

        try:
            import ApplicationServices as AS
            from AppKit import NSApp, NSApplicationActivationPolicyRegular

            # Check trust and warn clearly instead of silently failing
            trusted = AS.AXIsProcessTrusted()
            if not trusted:
                print(
                    "WARNING: Process not trusted — window cannot appear above fullscreen apps."
                )
                print(
                    "Fix: System Settings → Privacy & Security → Accessibility → add Terminal"
                )

            # Promote the process to a foreground app so macOS lets it
            # take focus from fullscreen apps. Without this, on Sonoma,
            # a bare Python process is treated as background-only and
            # cannot receive focus regardless of window level.
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)

            for ns_win in NSApp.windows():
                # canJoinAllSpaces (1) | NSWindowCollectionBehaviorStationary (16)
                # | fullScreenAuxiliary (256) = 273
                ns_win.setCollectionBehavior_(1 | 16 | 256)
                ns_win.setLevel_(1000)
                ns_win.makeKeyAndOrderFront_(None)
                ns_win.orderFrontRegardless()

            NSApp.activateIgnoringOtherApps_(True)
        except Exception as e:
            print(f"OVERLAY ERROR: {e}")

    def _build_ui(self) -> None:
        root = GridBackgroundWidget()
        self.setCentralWidget(root)

        page_layout = QVBoxLayout(root)
        page_layout.setContentsMargins(0, 0, 0, 40)
        page_layout.setSpacing(10)
        self.page_layout = page_layout

        self.content_column = QWidget()
        self.content_column.setMinimumWidth(930)
        self.content_column.setMaximumWidth(930)
        page_layout.addWidget(self.content_column, 0, Qt.AlignTop | Qt.AlignHCenter)

        content_layout = QVBoxLayout(self.content_column)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        self.content_layout = content_layout

        self.header = self._build_header()
        content_layout.addWidget(self.header, 0, Qt.AlignTop)

        self.results_shell = QFrame()
        self.results_shell.setObjectName("shell")
        self.results_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.results_shell.setMaximumHeight(560)
        self.results_shell.hide()
        content_layout.addWidget(self.results_shell, 0, Qt.AlignTop)
        page_layout.addStretch(1)

        shell_layout = QVBoxLayout(self.results_shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(18)

        shell_layout.addWidget(self._build_top_panels(), 3)
        self.entity_list = EntityList(open_handler=self.open_source_file)
        self.entity_list.list_widget.installEventFilter(self)
        shell_layout.addWidget(self.entity_list, 2)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 10, 22, 10)
        layout.setSpacing(18)

        icon = QLabel("⌕")
        icon.setObjectName("searchIcon")
        layout.addWidget(icon)

        self.query_input = QLineEdit()
        self.query_input.setObjectName("queryInput")
        self.query_input.setPlaceholderText("Search memory...")
        self.query_input.returnPressed.connect(self.run_search)
        self.query_input.installEventFilter(self)
        layout.addWidget(self.query_input, 1)

        self.cancel_label = QLabel("esc")
        self.cancel_label.setObjectName("cancelLabel")
        layout.addWidget(self.cancel_label)

        return header

    def _build_top_panels(self) -> QWidget:
        wrapper = QFrame()
        wrapper.setObjectName("topPanels")
        wrapper.setMinimumHeight(280)
        wrapper.setMaximumHeight(340)
        layout = QGridLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(0)

        self.files_panel = MemoryPanel("FILES", open_handler=self.open_source_file)
        self.matches_panel = MemoryPanel(
            "TOP_MATCHES", open_handler=self.open_source_file
        )
        self.metadata_panel = MemoryPanel(
            "METADATA", open_handler=self.open_source_file
        )
        self._bundle_panels = [
            self.files_panel,
            self.matches_panel,
            self.metadata_panel,
        ]

        for panel in self._bundle_panels:
            panel.list_widget.installEventFilter(self)

        layout.addWidget(self.files_panel, 0, 0)
        layout.addWidget(self.matches_panel, 0, 1)
        layout.addWidget(self.metadata_panel, 0, 2)
        return wrapper

    def _apply_styles(self) -> None:
        font_family = self._load_ui_font_family()
        self._font_family = font_family
        font = QFont(font_family)
        font.setStyleHint(QFont.SansSerif)
        QApplication.instance().setFont(font)

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: transparent;
                font-family: "{font_family}";
            }}
            #shell {{
                background: transparent;
                border: none;
            }}
            #header {{
                background: #0d0d0d;
                border: 1.5px solid {BOUNDARY_COLOR};
                border-top: none;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 16px;
                border-bottom-right-radius: 16px;
            }}
            #searchIcon {{
                color: {ACCENT_COLOR};
                font-family: "{font_family}";
                font-size: 26px;
                font-weight: 600;
            }}
            #queryInput {{
                background: transparent;
                border: none;
                color: {TEXT_PRIMARY};
                font-family: "{font_family}";
                font-size: 22px;
                font-weight: 400;
                padding: 6px 8px 6px 0;
            }}
            #queryInput::placeholder {{
                color: {TEXT_MUTED};
                font-family: "{font_family}";
            }}
            #cancelLabel {{
                color: #3d3d3d;
                font-family: "{font_family}";
                font-size: 13px;
                letter-spacing: 1px;
            }}
            #topPanels {{
                background: transparent;
                border: none;
            }}
            #memoryPanel {{
                background: #0f0f0f;
                border: 1.5px solid {BOUNDARY_COLOR};
                border-radius: 14px;
            }}
            #entityPanel {{
                background: #0f0f0f;
                border: 1.5px solid {BOUNDARY_COLOR};
                border-radius: 14px;
            }}
            #panelTitle {{
                color: {TEXT_MUTED};
                font-family: "{font_family}";
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 2px;
            }}
            #panelList {{
                background: transparent;
                color: {TEXT_PRIMARY};
                font-family: "{font_family}";
                font-size: 14px;
                outline: none;
                border-radius: 0px;
            }}
            #panelList::item {{
                padding: 7px 4px;
                border: none;
                border-radius: 6px;
            }}
            #panelList::item:selected {{
                background: {ACCENT_FAINT};
                color: {ACCENT_COLOR};
            }}
            #panelList::item:hover {{
                background: rgba(255, 255, 255, 0.03);
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                margin: 2px 1px 2px 1px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(90, 143, 214, 0.4);
                min-height: 24px;
                border-radius: 3px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            #panelFooter {{
                color: {TEXT_MUTED};
                font-family: "{font_family}";
                font-size: 12px;
                font-weight: 500;
                border-top: 1px solid #222222;
                padding-top: 8px;
            }}
            #entityHeader {{
                background: #0d0d0d;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom: 1px solid #1e1e1e;
            }}
            #entityHeaderTitle {{
                color: {TEXT_PRIMARY};
                font-family: "{font_family}";
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            #entityHeaderFilter {{
                color: {TEXT_MUTED};
                font-family: "{font_family}";
                font-size: 12px;
            }}
            #entityList {{
                background: transparent;
                color: {TEXT_PRIMARY};
                font-family: "{font_family}";
                font-size: 14px;
                outline: none;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            #entityList::item {{
                padding: 12px 14px;
                border-bottom: 1px solid #1a1a1a;
                border-radius: 0px;
            }}
            #entityList::item:selected {{
                background: {ACCENT_FAINT};
                color: {ACCENT_COLOR};
            }}
            #entityList::item:hover {{
                background: rgba(255, 255, 255, 0.03);
            }}
            #entityFooter {{
                background: #0f0f0f;
                border-top: 1px solid #1e1e1e;
                border-bottom-left-radius: 14px;
                border-bottom-right-radius: 14px;
            }}
            """
        )

    def run_search(self) -> None:
        prompt = self.query_input.text().strip()
        if not prompt:
            return
        if prompt.lower() == "exit!":
            QApplication.quit()
            return

        try:
            raw_matches = search_similar(
                prompt,
                k=20,
                with_vectors=True,
                score_threshold=0.2,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Search failed", str(exc))
            return

        self._results = raw_matches
        bundles = build_bundles(
            raw_matches,
            score_threshold=0.45,
            grouping_threshold=0.60,
            max_pool_size=20,
        )
        self._update_from_search(raw_matches, bundles)
        self.results_shell.show()
        if not self._results_visible:
            self._results_visible = True
            self._apply_search_compact_mode()
        QTimer.singleShot(0, self.play_results_animation)

    def _update_from_search(
        self,
        raw_matches: list[SearchResult],
        bundles: list[SearchBundle],
    ) -> None:
        top_bundles = bundles[:3]
        panels = [self.files_panel, self.matches_panel, self.metadata_panel]
        for index, panel in enumerate(panels):
            panel.title_label.setText(
                f"Bundle {index + 1}" if index < len(top_bundles) else "No bundle"
            )
            panel.set_bundle(top_bundles[index] if index < len(top_bundles) else None)
        if self._bundle_panels:
            self._last_bundle_focus_index = 0
            self._focus_list_widget(self._bundle_panels[0].list_widget, 0)

        entity_rows = self._build_entity_rows(raw_matches)
        self.entity_list.set_rows(
            entity_rows or [EntityRow("NO_RECOGNIZED_ENTITIES", "/", "")]
        )
        self._clear_inactive_selection(except_widget=self._bundle_panels[0].list_widget)

    def open_source_file(self, source_path: str) -> None:
        target = self._resolve_source_path(source_path)
        if not target.exists():
            QMessageBox.warning(
                self, "File not found", f"Could not find:\n{source_path}"
            )
            return

        try:
            self._launch_path(target)
        except Exception as exc:
            QMessageBox.critical(
                self, "Open failed", f"Could not open:\n{source_path}\n\n{exc}"
            )

    def _launch_path(self, target: Path) -> None:
        errors: list[str] = []

        system = platform.system()

        if system == "Windows":
            try:
                os.startfile(str(target))
                return
            except Exception as exc:
                errors.append(f"os.startfile failed: {exc}")

        if system == "Darwin":
            for command in (["open", str(target)],):
                try:
                    subprocess.Popen(command)
                    return
                except Exception as exc:
                    errors.append(f"{command[0]} failed: {exc}")

        if system == "Linux":
            if self._is_wsl():
                windows_target = self._to_windows_path(target)
                wsl_commands = [
                    ["wslview", str(target)],
                    ["/mnt/c/Windows/explorer.exe", windows_target],
                    ["explorer.exe", windows_target],
                    ["cmd.exe", "/C", "start", "", windows_target],
                ]
                for command in wsl_commands:
                    executable = command[0]
                    if (
                        executable not in {"/mnt/c/Windows/explorer.exe"}
                        and shutil.which(executable) is None
                    ):
                        errors.append(f"{executable} not found")
                        continue
                    try:
                        subprocess.Popen(command)
                        return
                    except Exception as exc:
                        errors.append(f"{executable} failed: {exc}")

            linux_commands = [
                ["xdg-open", str(target)],
                ["gio", "open", str(target)],
            ]
            for command in linux_commands:
                executable = command[0]
                if shutil.which(executable) is None:
                    errors.append(f"{executable} not found")
                    continue
                try:
                    subprocess.Popen(command)
                    return
                except Exception as exc:
                    errors.append(f"{executable} failed: {exc}")

        raise RuntimeError(
            f"Unsupported platform opener for {system}.\n" + "\n".join(errors)
        )

    def _is_wsl(self) -> bool:
        return (
            "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ
        )

    def _resolve_source_path(self, source_path: str) -> Path:
        raw = Path(source_path).expanduser()
        if raw.exists():
            return raw

        if self._is_wsl() and self._looks_like_windows_path(source_path):
            converted = self._windows_to_wsl_path(source_path)
            if converted.exists():
                return converted

        return raw

    def _looks_like_windows_path(self, source_path: str) -> bool:
        return (
            len(source_path) > 2
            and source_path[1] == ":"
            and source_path[2] in {"\\", "/"}
        )

    def _windows_to_wsl_path(self, source_path: str) -> Path:
        drive = source_path[0].lower()
        remainder = source_path[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / remainder

    def _to_windows_path(self, target: Path) -> str:
        try:
            completed = subprocess.run(
                ["wslpath", "-w", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip() or str(target)
        except Exception:
            return str(target)

    def _build_entity_rows(self, raw_matches: list[SearchResult]) -> list[EntityRow]:
        rows: list[EntityRow] = []
        seen: set[tuple[str, str, str]] = set()
        for result in raw_matches:
            payload = result.payload or {}
            file_name = str(payload.get("file_name") or "Unknown")
            source_path = Path(str(payload.get("source_path") or "/"))
            label = file_name
            location = str(source_path.parent).replace("\\", "/")
            resolved_source_path = str(source_path)
            key = (label, location, resolved_source_path)
            if key in seen:
                continue
            seen.add(key)
            rows.append(EntityRow(label, location, resolved_source_path))
            if len(rows) == 6:
                break
        return rows

    def eventFilter(self, watched: object, event: object) -> bool:
        if hasattr(event, "type") and event.type() == event.Type.KeyPress:
            key = event.key()
            if key in {Qt.Key_Return, Qt.Key_Enter}:
                if isinstance(watched, QListWidget):
                    item = watched.currentItem()
                    if item:
                        source_path = item.data(Qt.UserRole)
                        if isinstance(source_path, str) and source_path:
                            self.open_source_file(source_path)
                            return True
            if key in {Qt.Key_Left, Qt.Key_Right}:
                direction = -1 if key == Qt.Key_Left else 1
                if self._move_bundle_focus(watched, direction):
                    return True
            if (
                key == Qt.Key_Down
                and watched is self.query_input
                and self._move_focus_from_search_to_bundles()
            ):
                return True
            if key == Qt.Key_Down and self._move_focus_to_entities(watched):
                return True
            if key == Qt.Key_Up and self._move_focus_to_search(watched):
                return True
            if key == Qt.Key_Up and self._move_focus_to_bundles(watched):
                return True
        return super().eventFilter(watched, event)

    def _move_bundle_focus(self, watched: object, direction: int) -> bool:
        widgets = [panel.list_widget for panel in self._bundle_panels]
        if watched not in widgets:
            return False

        current_index = widgets.index(watched)
        target_widget = self._find_next_nonempty_bundle_widget(current_index, direction)
        if target_widget is None:
            return True

        self._last_bundle_focus_index = widgets.index(target_widget)
        self._focus_list_widget(
            target_widget,
            target_widget.currentRow() if target_widget.currentRow() >= 0 else 0,
        )
        return True

    def _move_focus_to_entities(self, watched: object) -> bool:
        widgets = [panel.list_widget for panel in self._bundle_panels]
        if watched not in widgets:
            return False

        current_widget = watched
        if current_widget.count() == 0:
            return False

        if current_widget.currentRow() < current_widget.count() - 1:
            return False

        self._last_bundle_focus_index = widgets.index(current_widget)
        if self.entity_list.list_widget.count() > 0:
            self._focus_list_widget(self.entity_list.list_widget, 0)
            return True
        return False

    def _move_focus_from_search_to_bundles(self) -> bool:
        if not self._results_visible or not self._bundle_panels:
            return False

        target_widget = self._bundle_panels[0].list_widget
        if target_widget.count() <= 0:
            fallback = self._find_next_nonempty_bundle_widget(-1, 1)
            if fallback is None:
                return False
            target_widget = fallback
        target_index = [panel.list_widget for panel in self._bundle_panels].index(
            target_widget
        )
        self._last_bundle_focus_index = target_index
        self._focus_list_widget(target_widget, 0)
        return True

    def _move_focus_to_search(self, watched: object) -> bool:
        widgets = [panel.list_widget for panel in self._bundle_panels]
        if watched not in widgets:
            return False

        current_widget = watched
        if current_widget.count() == 0:
            return False

        if current_widget.currentRow() > 0:
            return False

        self._last_bundle_focus_index = widgets.index(current_widget)
        self._clear_inactive_selection(except_widget=self.query_input)
        self.query_input.setFocus()
        return True

    def _move_focus_to_bundles(self, watched: object) -> bool:
        if watched is not self.entity_list.list_widget:
            return False

        if self.entity_list.list_widget.count() == 0:
            return False

        if self.entity_list.list_widget.currentRow() > 0:
            return False

        if not self._bundle_panels:
            return False

        target_index = min(self._last_bundle_focus_index, len(self._bundle_panels) - 1)
        target_widget = self._bundle_panels[target_index].list_widget
        if target_widget.count() == 0:
            fallback = self._find_next_nonempty_bundle_widget(target_index, -1)
            if fallback is None:
                fallback = self._find_next_nonempty_bundle_widget(target_index, 1)
            if fallback is None:
                return False
            target_widget = fallback
            target_index = [panel.list_widget for panel in self._bundle_panels].index(
                target_widget
            )
        self._last_bundle_focus_index = target_index
        self._focus_list_widget(target_widget, target_widget.count() - 1)
        return True

    def _find_next_nonempty_bundle_widget(
        self, start_index: int, direction: int
    ) -> QListWidget | None:
        widgets = [panel.list_widget for panel in self._bundle_panels]
        target_index = start_index + direction
        while 0 <= target_index < len(widgets):
            if widgets[target_index].count() > 0:
                return widgets[target_index]
            target_index += direction
        return None

    def _clear_inactive_selection(self, except_widget: QListWidget) -> None:
        widgets = [panel.list_widget for panel in self._bundle_panels] + [
            self.entity_list.list_widget
        ]
        for widget in widgets:
            if widget is except_widget:
                continue
            widget.clearSelection()

    def _focus_list_widget(self, widget: QListWidget, row: int) -> None:
        self._clear_inactive_selection(except_widget=widget)
        widget.setFocus()
        if widget.count() <= 0:
            return
        safe_row = max(0, min(row, widget.count() - 1))
        widget.setCurrentRow(safe_row, QItemSelectionModel.ClearAndSelect)

    def _apply_search_compact_mode(self) -> None:
        # Width stays the same; no font/size change between pre and post search
        pass

    def _load_ui_font_family(self) -> str:
        candidate_paths: list[str] = []
        if UI_FONT_FILE:
            candidate_paths.append(UI_FONT_FILE)
        candidate_paths.extend(UI_FONT_FALLBACKS)

        for candidate in candidate_paths:
            if not candidate:
                continue
            font_path = Path(candidate).expanduser()
            if not font_path.exists():
                continue
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id == -1:
                continue
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                return families[0]

        return UI_FONT_FAMILY

    def play_intro_animation(self) -> None:
        self._animate_fly_in(self.header, distance=64, duration=620)

    def play_results_animation(self) -> None:
        self._animate_fly_in(self.results_shell, distance=52, duration=520)

    def _animate_fly_in(self, widget: QWidget, *, distance: int, duration: int) -> None:
        end_geometry = widget.geometry()
        if end_geometry.isNull():
            return

        start_geometry = QRect(end_geometry)
        start_geometry.translate(0, -distance)

        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)

        geometry_anim = QPropertyAnimation(widget, b"geometry", self)
        geometry_anim.setDuration(duration)
        geometry_anim.setStartValue(start_geometry)
        geometry_anim.setEndValue(end_geometry)
        geometry_anim.setEasingCurve(QEasingCurve.OutCubic)

        opacity_anim = QPropertyAnimation(effect, b"opacity", self)
        opacity_anim.setDuration(duration)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(geometry_anim)
        group.addAnimation(opacity_anim)

        def cleanup() -> None:
            widget.setGeometry(end_geometry)
            effect.setOpacity(1.0)
            if group in self._active_animations:
                self._active_animations.remove(group)

        group.finished.connect(cleanup)
        self._active_animations.append(group)
        group.start()


def launch_desktop_app(*, show_on_start: bool = True) -> int:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow(show_on_start=show_on_start)
    app._main_window = window
    window.show()
    if not show_on_start:
        QTimer.singleShot(0, window.hide)
    return app.exec()
