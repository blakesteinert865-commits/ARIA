"""Aria Tools desktop interface.

The GUI is deliberately lightweight: the model sees only the latest three
conversation messages, conversations are not saved to disk, and backend work
runs in a QThread so the interface remains responsive.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from collections import deque
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

try:
    import mistune
except ImportError:  # The app still works without third-party Markdown packages.
    mistune = None

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR  # application, bridge, tools and writable data share one folder
BACKEND_SCRIPT = BACKEND_DIR / "crutch.py"
BACKEND_BRIDGE = APP_DIR / "aria_backend_bridge.py"
MEMORY_MESSAGE_LIMIT = 3


APP_STYLESHEET = """
QWidget#appRoot {
    color: #dce8ff;
    font-family: 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
    font-size: 14px;
    background-color: #06152b;
}
QLabel { background: transparent; }
QFrame#header {
    background-color: rgba(4, 15, 33, 188);
    border: 1px solid rgba(85, 145, 232, 95);
    border-radius: 12px;
}
QLabel#appTitle { color: #f0f6ff; font-size: 17px; font-weight: 700; }
QLabel#appSubtitle { color: #91acd2; font-size: 11px; }
QLabel#statusPill {
    color: #b8f7d0;
    background-color: rgba(24, 84, 67, 175);
    border: 1px solid rgba(90, 215, 162, 90);
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 600;
}
QScrollArea#chatScroll { border: none; background: transparent; }
QWidget#messageCanvas { background: transparent; }
QFrame#userBubble {
    background-color: rgba(25, 56, 95, 230);
    border: 1px solid rgba(88, 156, 255, 155);
    border-radius: 12px;
}
QFrame#assistantBubble {
    background-color: rgba(5, 17, 35, 228);
    border: 1px solid rgba(66, 119, 194, 145);
    border-radius: 12px;
}
QLabel#userSpeaker { color: #9bc7ff; font-weight: 700; }
QLabel#assistantSpeaker { color: #80d5ff; font-weight: 700; }
QLabel#typingLabel { color: #a5c7f4; font-style: italic; }
QTextBrowser#messageText {
    background: transparent;
    border: none;
    color: #e3edff;
    padding: 0;
    selection-background-color: #285ca8;
}
QLineEdit#promptInput {
    background-color: rgba(10, 27, 50, 235);
    color: #ffffff;
    border: 1px solid rgba(82, 144, 235, 150);
    border-radius: 10px;
    padding: 11px 12px;
    selection-background-color: #285ca8;
}
QLineEdit#promptInput:focus { border: 1px solid #64a9ff; }
QPushButton {
    background-color: #347fe5;
    color: #ffffff;
    border: 1px solid rgba(150, 198, 255, 70);
    border-radius: 8px;
    padding: 9px 13px;
    font-weight: 600;
}
QPushButton:hover { background-color: #4a94f5; }
QPushButton:pressed { background-color: #2867bd; }
QPushButton:disabled { color: #8394ad; background-color: rgba(35, 54, 80, 190); }
QPushButton#copyButton {
    background-color: rgba(26, 49, 79, 210);
    color: #a9cfff;
    border: 1px solid rgba(92, 143, 209, 95);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
}
QPushButton#copyButton:hover { background-color: rgba(43, 77, 119, 235); }
QPushButton#clearButton {
    background-color: rgba(32, 54, 82, 230);
    border: 1px solid rgba(111, 153, 208, 100);
}
QPushButton#clearButton:hover { background-color: rgba(48, 76, 111, 245); }
QPushButton#exitButton {
    background-color: rgba(116, 44, 65, 235);
    border: 1px solid rgba(255, 170, 190, 70);
}
QPushButton#exitButton:hover { background-color: #a63e59; }
QLabel#footerHint { color: #829dbf; font-size: 11px; }
"""


def _inline_markdown_fallback(text: str) -> str:
    """Small safe inline-Markdown renderer used when Mistune isn't installed."""
    escaped = html.escape(text, quote=False)
    # Inline code first, so its contents don't get bold/italic substitutions.
    code_chunks: list[str] = []

    def hold_code(match: re.Match[str]) -> str:
        code_chunks.append(f'<code style="color:#b9d8ff;background-color:#101f35;">{match.group(1)}</code>')
        return f"\x00CODE{len(code_chunks) - 1}\x00"

    escaped = re.sub(r"`([^`]+)`", hold_code, escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"~~(.+?)~~", r"<s>\1</s>", escaped)

    def replace_link(match: re.Match[str]) -> str:
        label, url = match.group(1), html.unescape(match.group(2))
        if url.startswith(("https://", "http://", "mailto:")):
            return f'<a href="{html.escape(url, quote=True)}">{label}</a>'
        return label

    escaped = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", replace_link, escaped)
    for index, fragment in enumerate(code_chunks):
        escaped = escaped.replace(f"\x00CODE{index}\x00", fragment)
    return escaped


def markdown_to_html(source: str) -> str:
    """Convert Markdown to HTML while escaping raw HTML from the model/user."""
    if mistune is not None:
        try:
            renderer = mistune.create_markdown(escape=True, plugins=["table", "strikethrough"])
            rendered = renderer(source)
            return rendered
        except Exception:
            pass

    # A compact fallback supporting headings, paragraphs, lists and fenced code.
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    code_lines: list[str] | None = None

    def flush_paragraph() -> None:
        if paragraph:
            output.append("<p>" + "<br>".join(_inline_markdown_fallback(line) for line in paragraph) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.append(f"</{list_kind}>")
            list_kind = None

    for line in lines:
        if line.strip().startswith("```"):
            if code_lines is None:
                flush_paragraph()
                close_list()
                code_lines = []
            else:
                output.append(
                    '<pre style="white-space:pre-wrap;background-color:#101f35;color:#c9e2ff;'
                    'padding:8px;border-radius:6px;"><code>'
                    + html.escape("\n".join(code_lines))
                    + "</code></pre>"
                )
                code_lines = None
            continue
        if code_lines is not None:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{_inline_markdown_fallback(heading.group(2))}</h{level}>")
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph()
            desired = "ul" if bullet else "ol"
            if list_kind != desired:
                close_list()
                output.append(f"<{desired}>")
                list_kind = desired
            item_text = (bullet or numbered).group(1)
            output.append(f"<li>{_inline_markdown_fallback(item_text)}</li>")
            continue
        close_list()
        paragraph.append(line)

    flush_paragraph()
    close_list()
    if code_lines is not None:
        output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    return "\n".join(output) or "<p></p>"


def _message_document(source: str) -> str:
    body = markdown_to_html(source)
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        body { color:#e3edff; font-family:'Segoe UI','Noto Sans',Arial,sans-serif;
               font-size:14px; margin:0; padding:0; }
        p { margin:2px 0 7px 0; }
        p:last-child { margin-bottom:0; }
        h1,h2,h3,h4,h5,h6 { color:#9ccaff; margin:7px 0 4px 0; }
        a { color:#72baff; }
        code { color:#b9d8ff; background-color:#101f35; }
        pre { white-space:pre-wrap; background-color:#101f35; color:#c9e2ff;
              padding:8px; border-radius:6px; }
        blockquote { border-left:3px solid #478de3; margin-left:0; padding-left:10px; color:#a9c0df; }
        table { border-collapse:collapse; }
        th,td { border:1px solid #365577; padding:4px 7px; }
        hr { border:0; border-top:1px solid #365577; }
    </style></head><body>""" + body + "</body></html>"""


class MessageTextBrowser(QTextBrowser):
    """A rich-text message area whose height follows its Markdown document."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("messageText")
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("QTextBrowser { background:transparent; border:none; padding:0; }")

    def set_message(self, text: str) -> None:
        self.setHtml(_message_document(text))
        self._fit_document()

    def _fit_document(self) -> None:
        width = max(160, self.viewport().width())
        self.document().setTextWidth(width)
        height = int(self.document().documentLayout().documentSize().height()) + 8
        self.setFixedHeight(max(26, height))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        super().resizeEvent(event)
        self._fit_document()


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        is_user = role == "user"
        self.setObjectName("userBubble" if is_user else "assistantBubble")
        self.raw_text = text
        self.setMaximumWidth(820)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        bubble_layout = QVBoxLayout(self)
        bubble_layout.setContentsMargins(12, 9, 12, 11)
        bubble_layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        speaker = QLabel("You" if is_user else "Aria")
        speaker.setObjectName("userSpeaker" if is_user else "assistantSpeaker")
        header.addWidget(speaker)
        header.addStretch(1)
        if not is_user:
            self.copy_btn = QPushButton("Copy")
            self.copy_btn.setObjectName("copyButton")
            self.copy_btn.setToolTip("Copy this response")
            self.copy_btn.clicked.connect(self.copy_response)
            header.addWidget(self.copy_btn)
        bubble_layout.addLayout(header)

        self.text_browser = MessageTextBrowser(self)
        self.text_browser.set_message(text)
        bubble_layout.addWidget(self.text_browser)

    def set_text(self, text: str) -> None:
        self.raw_text = text
        self.text_browser.set_message(text)

    def copy_response(self) -> None:
        QApplication.clipboard().setText(self.raw_text)
        if hasattr(self, "copy_btn"):
            self.copy_btn.setText("Copied")
            QTimer.singleShot(1200, lambda: self.copy_btn.setText("Copy") if self.copy_btn else None)


class TypingBubble(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("assistantBubble")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(13, 9, 13, 9)
        self.label = QLabel("Aria is thinking…")
        self.label.setObjectName("typingLabel")
        layout.addWidget(self.label)
        self.setMaximumWidth(260)


class BackendWorker(QThread):
    """Stream JSON-lines from the bridge without blocking the Qt event loop."""
    chunk_ready = pyqtSignal(str)

    def __init__(self, prompt: str, context_json: str, generation: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.prompt = prompt
        self.context_prompt = context_json
        self.generation = generation
        self.result_text = ""

    def run(self) -> None:
        try:
            # Avoid stderr/stdout pipe deadlocks and prevent diagnostics from
            # contaminating the structured stream. The file is auto-removed.
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as error_log:
                with subprocess.Popen(
                    [sys.executable, "-u", str(BACKEND_BRIDGE), "--stream-events",
                     "--prompt", self.prompt, "--context", self.context_prompt],
                    cwd=str(BACKEND_DIR), stdout=subprocess.PIPE, stderr=error_log,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                ) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        try:
                            event = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(event, dict) or not isinstance(event.get("text"), str):
                            continue
                        if event.get("type") == "delta":
                            self.chunk_ready.emit(event["text"])
                        elif event.get("type") == "final":
                            self.result_text = event["text"]
                    exit_code = process.wait()
                if exit_code != 0:
                    error_log.seek(0)
                    detail = error_log.read()[-1800:].strip() or f"Exit code {exit_code}"
                    self.result_text = f"Script Error: {detail}"
            if not self.result_text:
                self.result_text = "(No output returned.)"
        except Exception as exc:
            self.result_text = f"System Error: {exc}"


class CachyChatBot(QWidget):
    def __init__(self):
        super().__init__()
        self.background_pixmap = QPixmap(str(APP_DIR / "Arialogobase.png"))
        self.glow_pixmap = QPixmap(str(APP_DIR / "AriaGlow.png"))
        self.memory: deque[tuple[str, str]] = deque(maxlen=MEMORY_MESSAGE_LIMIT)
        self.context_generation = 0
        self.worker: BackendWorker | None = None
        self.typing_bubble: TypingBubble | None = None
        self.stream_bubble: MessageBubble | None = None
        self.stream_text = ""
        self._glow_phase = 0.0
        self._glow_opacity = 0.31
        self._init_ui()

        # A slow, restrained pulse only changes the separate blue underglow layer.
        self.glow_timer = QTimer(self)
        self.glow_timer.setInterval(100)
        self.glow_timer.timeout.connect(self._pulse_glow)
        self.glow_timer.start()

    def _init_ui(self) -> None:
        self.setObjectName("appRoot")
        self.setWindowTitle("Aria Tools")
        self.setWindowIcon(QIcon(str(APP_DIR / "AriaAppIcon.png")))
        self.resize(760, 820)
        self.setMinimumSize(440, 560)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(APP_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(13, 13, 13, 10)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(13, 10, 13, 10)
        header_layout.setSpacing(10)

        icon_label = QLabel()
        icon_pixmap = QPixmap(str(APP_DIR / "AriaAppIcon.png"))
        if not icon_pixmap.isNull():
            icon_label.setPixmap(icon_pixmap.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        icon_label.setFixedSize(36, 36)
        header_layout.addWidget(icon_label)

        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        title = QLabel("Aria")
        title.setObjectName("appTitle")
        subtitle = QLabel("Tools first · compact context")
        subtitle.setObjectName("appSubtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column)
        header_layout.addStretch(1)

        self.status_label = QLabel("● Ready")
        self.status_label.setObjectName("statusPill")
        self.status_label.setToolTip("Backend and task status. Enter /status for details.")
        header_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(header)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_scroll.viewport().setStyleSheet("background:transparent;")

        self.message_canvas = QWidget()
        self.message_canvas.setObjectName("messageCanvas")
        self.messages_layout = QVBoxLayout(self.message_canvas)
        self.messages_layout.setContentsMargins(3, 3, 3, 8)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)
        self.chat_scroll.setWidget(self.message_canvas)
        root.addWidget(self.chat_scroll, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(7)
        self.input_field = QLineEdit()
        self.input_field.setObjectName("promptInput")
        self.input_field.setPlaceholderText("Ask Aria…  (Enter to send, /help for commands)")
        self.input_field.returnPressed.connect(self.handle_submit)
        controls.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self.handle_submit)
        controls.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearButton")
        self.clear_btn.setToolTip("Clear the visible conversation and short-term context (Ctrl+L)")
        self.clear_btn.clicked.connect(self.clear_chat)
        controls.addWidget(self.clear_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setObjectName("exitButton")
        self.exit_btn.setToolTip("Close Aria Tools (Esc)")
        self.exit_btn.clicked.connect(self.close)
        controls.addWidget(self.exit_btn)
        root.addLayout(controls)

        footer = QLabel("/status · /debug · /help · /clear     |     Context: last 3 messages only · not saved")
        footer.setObjectName("footerHint")
        root.addWidget(footer)

        self._add_message(
            "assistant",
            "Aria is ready.\n\n"
            "Type `/help` to see commands or `/status` to check the backend.",
        )
        self.input_field.setFocus()
        self._update_status()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#06152b"))

        if not self.background_pixmap.isNull():
            background = self.background_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - background.width()) // 2
            y = (self.height() - background.height()) // 2
            painter.setOpacity(0.25)
            painter.drawPixmap(x, y, background)
            painter.setOpacity(1.0)
            painter.fillRect(self.rect(), QColor(3, 10, 24, 48))

        if not self.glow_pixmap.isNull():
            glow = self.glow_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - glow.width()) // 2
            y = (self.height() - glow.height()) // 2
            painter.setOpacity(self._glow_opacity)
            painter.drawPixmap(x, y, glow)
        painter.end()

    def _pulse_glow(self) -> None:
        # Increment phase slowly: just a small underglow breath, not a flashing logo.
        import math

        self._glow_phase += 0.075
        self._glow_opacity = 0.29 + (math.sin(self._glow_phase) * 0.035)
        self.update()

    def _set_status(self, text: str, state: str = "ready") -> None:
        palette = {
            "ready": ("#b8f7d0", "rgba(24, 84, 67, 175)", "rgba(90, 215, 162, 90)"),
            "busy": ("#b9d8ff", "rgba(29, 66, 112, 190)", "rgba(95, 164, 255, 120)"),
            "warning": ("#ffd29b", "rgba(105, 69, 28, 190)", "rgba(235, 169, 89, 130)"),
            "error": ("#ffb4c3", "rgba(104, 35, 55, 190)", "rgba(240, 101, 134, 135)"),
        }
        foreground, background, border = palette.get(state, palette["ready"])
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(
            "QLabel#statusPill {"
            f"color:{foreground};background-color:{background};border:1px solid {border};"
            "border-radius:9px;padding:5px 9px;font-size:11px;font-weight:600;}"
        )

    def _update_status(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._set_status("Working…", "busy")
        elif not BACKEND_SCRIPT.is_file() or not BACKEND_BRIDGE.is_file():
            self._set_status("Backend missing", "warning")
        else:
            self._set_status("Ready", "ready")

    def _add_message(self, role: str, text: str) -> MessageBubble:
        row = QWidget(self.message_canvas)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        bubble = MessageBubble(role, text, row)
        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, Qt.AlignmentFlag.AlignRight)
        else:
            row_layout.addWidget(bubble, 0, Qt.AlignmentFlag.AlignLeft)
            row_layout.addStretch(1)

        # Keep a stretch at the end so short conversations remain top-aligned.
        self.messages_layout.insertWidget(max(0, self.messages_layout.count() - 1), row)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return bubble

    def _add_typing_bubble(self) -> None:
        if self.typing_bubble is not None:
            return
        row = QWidget(self.message_canvas)
        row.setObjectName("typingRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.typing_bubble = TypingBubble(row)
        row_layout.addWidget(self.typing_bubble, 0, Qt.AlignmentFlag.AlignLeft)
        row_layout.addStretch(1)
        self.messages_layout.insertWidget(max(0, self.messages_layout.count() - 1), row)
        self._typing_row = row
        self._scroll_to_bottom()

    def _remove_typing_bubble(self) -> None:
        row = getattr(self, "_typing_row", None)
        if row is not None:
            self.messages_layout.removeWidget(row)
            row.deleteLater()
        self.typing_bubble = None
        self._typing_row = None

    def _scroll_to_bottom(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear_chat(self) -> None:
        # Invalidate an in-flight reply as well as clearing visible and short-term context.
        # This never touches crutch.py's persistent notes, lists or timetable.
        self.context_generation += 1
        self.stream_bubble = None
        self.stream_text = ""
        self.memory.clear()
        self._remove_typing_bubble()
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.typing_bubble = None
        self._typing_row = None
        self._add_message("assistant", "Conversation cleared. Short-term context has been reset.")
        self.input_field.setFocus()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override name
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_L and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.clear_chat()
            return
        if event.key() == Qt.Key.Key_N and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.clear_chat()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        # Avoid destroying a live QThread if the window is closed mid-request.
        if self.worker is not None and self.worker.isRunning():
            self._set_status("Finishing request…", "busy")
            self.worker.wait()
        event.accept()

    def _build_context_json(self) -> str:
        """Role-separated *previous* turns for the model fallback.

        Do not include the current user prompt here: the bridge adds it as a
        separate final user message. Assistant entries include both Crutch
        tool answers and model replies, since both are displayed by this GUI.
        """
        return json.dumps(
            [
                {"role": role, "content": content}
                for role, content in self.memory
                if role in ("user", "assistant") and content.strip()
            ],
            ensure_ascii=False,
        )

    @staticmethod
    def _model_server_status() -> str:
        # Probe the OpenAI-compatible /models endpoint without submitting a prompt.
        base_url = os.environ.get("CRUTCH_URL", "http://localhost:1234/v1/chat/completions")
        parsed = urlsplit(base_url)
        if not parsed.scheme or not parsed.netloc:
            return "URL is invalid"
        path = re.sub(r"/chat/completions/?$", "/models", parsed.path)
        probe_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        try:
            with urlopen(Request(probe_url, method="GET"), timeout=0.7) as response:
                return f"Reachable (HTTP {response.status})"
        except Exception as exc:
            return f"Unreachable ({type(exc).__name__})"

    def _local_command(self, text: str) -> bool:
        command = text.strip().lower()
        if command == "/status":
            script_exists = BACKEND_SCRIPT.is_file()
            bridge_exists = BACKEND_BRIDGE.is_file()
            running = self.worker is not None and self.worker.isRunning()
            model_status = self._model_server_status()
            status = [
                "### Aria status",
                f"- **Backend script:** {'Found' if script_exists else 'Missing'} (`{BACKEND_SCRIPT}`)",
                f"- **GUI bridge:** {'Found' if bridge_exists else 'Missing'} (`{BACKEND_BRIDGE}`)",
                f"- **Model server:** {model_status}",
                f"- **Python:** `{sys.executable}`",
                f"- **Working directory:** `{BACKEND_DIR}`",
                f"- **Execution:** {'Running' if running else 'Idle'}",
                f"- **Short-term context:** {len(self.memory)}/{MEMORY_MESSAGE_LIMIT} messages",
                "- **Saved history:** Disabled",
                "- **File attachments:** Not enabled",
            ]
            self._add_message("assistant", "\n".join(status))
            if not script_exists:
                self._set_status("Backend missing", "warning")
            return True
        if command == "/debug":
            debug_path = Path(os.environ.get("ARIA_DEBUG_FILE", str(APP_DIR / "data" / "aria_debug.json"))).expanduser()
            try:
                record = json.loads(debug_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._add_message("assistant", "### Diagnostics\nNo request trace exists yet. Send a normal question first, then run `/debug`.")
                return True
            except (OSError, ValueError) as exc:
                self._add_message("assistant", f"### Diagnostics\nCould not read `{debug_path}`: `{type(exc).__name__}: {exc}`")
                return True
            fields = [
                ("Request", record.get("request_id")),
                ("Time (UTC)", record.get("timestamp_utc")),
                ("Question", record.get("question")),
                ("Route", record.get("route")),
                ("Matched tool", record.get("tool")),
                ("RAG eligible", record.get("rag_eligible")),
                ("RAG status", record.get("rag_status")),
                ("Article", record.get("article_title")),
                ("Article path", record.get("article_path")),
                ("Cache hit", record.get("cache_hit")),
                ("Passage characters", record.get("passage_chars")),
                ("Context attached to model prompt", record.get("context_attached")),
                ("Model endpoint", record.get("model_endpoint")),
                ("Model", record.get("model_name")),
                ("Model status", record.get("model_status")),
                ("Model time", f"{record.get('model_duration_ms')} ms" if record.get("model_duration_ms") is not None else None),
                ("Reply source", record.get("reply_source")),
                ("Fallback reason", record.get("fallback_reason")),
                ("Total time", f"{record.get('total_duration_ms')} ms" if record.get("total_duration_ms") is not None else None),
            ]
            lines = ["### Latest request diagnostics"]
            lines.extend(f"- **{label}:** {value}" for label, value in fields if value is not None)
            errors = record.get("errors") or []
            if errors:
                lines.append("- **Errors:** " + "; ".join(str(error) for error in errors))
            lines.append(f"\nFull trace: `{debug_path}`")
            self._add_message("assistant", "\n".join(lines))
            return True
        if command in ("/help", "/?"):
            self._add_message(
                "assistant",
                "### Commands\n"
                "- `/status` — inspect backend path, Python interpreter and short-term context.\n"
                "- `/debug` — show the latest routing, Wikipedia retrieval and model handoff trace.\n"
                "- `/clear` — clear the conversation and reset short-term context.\n"
                "- `/help` — show this command list.\n\n"
                "Aria remembers only the latest **3 messages** for model context. Nothing is saved to disk.",
            )
            return True
        if command in ("/clear", "/reset"):
            self.clear_chat()
            return True
        return False

    def handle_submit(self) -> None:
        user_text = self.input_field.text().strip()
        if not user_text:
            return
        self.input_field.clear()

        # Local commands must still work while the backend is busy.
        self._add_message("user", user_text)
        if user_text.startswith("/") and self._local_command(user_text):
            self.input_field.setFocus()
            self._scroll_to_bottom()
            return

        if self.worker is not None and self.worker.isRunning():
            self._add_message("assistant", "That message was **not sent** because Aria is still processing the previous request. Use `/status` or retry when it finishes.")
            self._scroll_to_bottom()
            self.input_field.setFocus()
            return

        # Snapshot the previous three turns BEFORE adding this latest user
        # message. The bridge supplies the latest prompt as its own user turn.
        # This avoids duplicating the current question or flattening speakers.
        context_prompt = self._build_context_json()
        self.memory.append(("user", user_text))

        self.send_btn.setEnabled(False)
        self._add_typing_bubble()
        self._set_status("Thinking…", "busy")

        self.worker = BackendWorker(user_text, context_prompt, self.context_generation, self)
        self.worker.chunk_ready.connect(self._on_backend_delta)
        self.worker.finished.connect(self._on_backend_finished)
        self.worker.start()
        self._scroll_to_bottom()

    def _on_backend_delta(self, fragment: str) -> None:
        worker = self.worker
        if worker is None or worker.generation != self.context_generation or not fragment:
            return
        if self.stream_bubble is None:
            self._remove_typing_bubble()
            self.stream_bubble = self._add_message("assistant", "")
        self.stream_text += fragment
        self.stream_bubble.set_text(self.stream_text)
        self._scroll_to_bottom()

    def _on_backend_finished(self) -> None:
        worker = self.worker
        if worker is None:
            return
        response = worker.result_text or "(No output returned.)"
        self._remove_typing_bubble()
        # Ignore stale streams after /clear; never append a second final bubble.
        if worker.generation == self.context_generation:
            self.memory.append(("assistant", response))
            if self.stream_bubble is None:
                self._add_message("assistant", response)
            else:
                self.stream_bubble.set_text(response)
        self.stream_bubble = None
        self.stream_text = ""
        worker.deleteLater()
        self.worker = None
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        if response.startswith(("System Error:", "Script Error:")):
            self._set_status("Backend error", "error")
        else:
            self._update_status()
        self._scroll_to_bottom()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Aria Tools")
    app.setOrganizationName("Aria")
    window = CachyChatBot()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
