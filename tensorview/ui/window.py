"""The main TensorView desktop window."""

import re
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..analysis import demo_files
from ..exporting import TENSOR_EXPORTS, atomic_destination, export_report, export_tensor
from ..loaders import FORMATS, open_file
from ..model import TensorViewError
from .explorer import Explorer
from .inspector import Inspector
from .jobs import Jobs
from .theme import ComboBox, button, label
from .widgets import row
from .workspaces import Comparison, Validation

FILE_FILTER = (
    f"Tensor and model files ({' '.join('*' + extension for extension in FORMATS)})"
    ";;All files (*)"
)
REPORT_FILTER = "JSON report (*.json);;CSV report (*.csv);;HTML report (*.html)"


def application_icon(size=128) -> QPixmap:
    colors = ("#263A53", "#54919B", "#66DCCA", "#456F91", "#81F1DA", "#BDF8DC", "#6A94DF", "#91D6D2", "#FFCB93")
    image = QPixmap(size, size)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 128, size / 128)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#223346"))
    painter.drawRoundedRect(QRectF(0, 0, 128, 128), 28, 28)
    for index, color in enumerate(colors):
        x, y = 23 + index % 3 * 29, 23 + index // 3 * 29
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(QRectF(x, y, 24, 24), 5, 5)
    painter.end()
    return image


class MainWindow(QMainWindow):
    def __init__(self, *, demo=True):
        super().__init__()
        self.setWindowTitle("TensorView")
        self.setWindowIcon(QIcon(application_icon()))
        self.resize(1360, 870)
        self.setMinimumSize(970, 680)
        self.setAcceptDrops(True)
        self.sources = {}
        self.selections = {}
        self.selected_key = None
        self.jobs = Jobs(self)
        self.jobs.error.connect(self.show_error)
        self.jobs.busy.connect(self._busy)
        self.close_pending = False

        self.inspector = Inspector(self.jobs)
        self.comparison = Comparison(self.jobs, self._open_for_combo)
        self.validation = Validation(self.jobs, self.show_tensor)

        workspace = QWidget()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(workspace)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setSpacing(11)
        header_layout.setContentsMargins(15, 10, 18, 10)
        logo = QLabel()
        logo.setPixmap(application_icon(35))
        header_layout.addWidget(logo)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addWidget(label("TensorView", "brand"))
        brand.addWidget(label("Tensor & checkpoint workbench", "caption"))
        header_layout.addLayout(brand)
        header_layout.addSpacing(32)
        self.navigation = QButtonGroup(self)
        self.navigation.setExclusive(True)
        for number, name in enumerate(("Inspect", "Compare", "Validate")):
            navigation = button(name, "nav")
            navigation.setCheckable(True)
            self.navigation.addButton(navigation, number)
            header_layout.addWidget(navigation)
        self.navigation.idClicked.connect(self._set_page)
        self.navigation.button(0).setChecked(True)
        header_layout.addStretch()
        self.export_button = button("Export…")
        self.export_button.setMenu(self._export_menu(self.export_button))
        header_layout.addWidget(self.export_button)
        self.open_button = button("+  Open file", "primary")
        self.open_button.clicked.connect(self.open_dialog)
        header_layout.addWidget(self.open_button)
        layout.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.explorer = Explorer()
        self.explorer.setFixedWidth(260)
        self.explorer.file_changed.connect(self._activate_file)
        self.explorer.tensor_changed.connect(self._activate_tensor)
        self.explorer.close_file.connect(self.close_file)
        self.explorer.open_file.connect(self.open_dialog)
        body.addWidget(self.explorer)
        self.pages = QStackedWidget()
        for page in (self.inspector, self.comparison, self.validation):
            self.pages.addWidget(page)
        body.addWidget(self.pages, 1)
        layout.addLayout(body, 1)

        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = row(margins=(15, 0, 17, 0), spacing=8)
        self.connection = label("●", "accent")
        self.status = label("Ready", "caption")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(108, 6)
        self.progress.hide()
        footer_layout.addWidget(self.connection)
        footer_layout.addWidget(self.status)
        footer_layout.addStretch()
        footer_layout.addWidget(self.progress)
        footer_layout.addWidget(label("Local processing", "caption"))
        footer.setLayout(footer_layout)
        footer.setFixedHeight(31)
        layout.addWidget(footer)

        self.notice = QTimer(self)
        self.notice.setInterval(3800)
        self.notice.setSingleShot(True)
        self.notice.timeout.connect(lambda: self.status.setText("Ready") if not self.progress.isVisible() else None)
        self._make_menus()
        if demo:
            self.load_demo()
        else:
            self._sync_sources()

    def _make_menus(self):
        bar = self.menuBar()
        file = bar.addMenu("&File")
        self._action(file, "Open files…", self.open_dialog, QKeySequence.StandardKey.Open)
        self._action(file, "Close selected file", self.close_file, QKeySequence.StandardKey.Close)
        self._action(file, "Load example checkpoints", self.load_demo)
        file.addSeparator()
        file.addMenu(self._export_menu(file, "Export"))
        file.addSeparator()
        self._action(file, "Quit", self.close, QKeySequence.StandardKey.Quit)
        view = bar.addMenu("&View")
        for index, name in enumerate(("Inspect", "Compare", "Validate")):
            self._action(view, name, lambda checked=False, index=index: self._set_page(index), f"Ctrl+{index + 1}")
        self._action(view, "Filter tensors", self.explorer.search.setFocus, "Ctrl+F")
        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "About TensorView", self._about)

    def _export_menu(self, parent, title=""):
        menu = QMenu(title, parent)
        self._action(menu, "Convert or extract tensor…", self._convert)
        self._action(menu, "Export current report…", self._save_report)
        self._action(menu, "Save current plot as PNG…", self._save_plot)
        self._action(menu, "Save inspector histogram as PNG…", lambda: self._save_plot(histogram=True))
        return menu

    def _action(self, menu, title, slot, shortcut=None):
        action = QAction(title, menu)
        action.triggered.connect(lambda: slot())
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)

    def _about(self):
        box = QMessageBox(self)
        box.setWindowTitle("About TensorView")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setIconPixmap(application_icon(72))
        box.setText(
            f"TensorView {__version__}\n\nInspect, compare, validate and convert local tensor files.\n"
            "Supports NumPy, SafeTensors, PyTorch, HDF5, ONNX and MATLAB.\n\n"
            "Pan by dragging a plot and zoom with your mouse wheel or trackpad."
        )
        box.exec()

    def _set_page(self, index):
        self.navigation.button(index).setChecked(True)
        self.pages.setCurrentIndex(index)

    def load_demo(self):
        self.sources.update(zip(("example:baseline", "example:tuned"), demo_files(), strict=True))
        self.comparison.invalidate()
        self.validation.invalidate()
        self.selected_key = "example:baseline"
        self.selections[self.selected_key] = "encoder.projection.weight"
        self._sync_sources()
        self._set_page(0)

    def _sync_sources(self):
        self.explorer.show_files(self.sources, self.selected_key)
        self.comparison.set_sources(self.sources)
        self.validation.set_sources(self.sources)
        self._activate_file(self.selected_key)

    def _activate_file(self, key):
        self.selected_key = key
        self.explorer.remove.setEnabled(key is not None)
        source = self.sources.get(key)
        selected = self.selections.get(key)
        if source is not None and selected not in source.tensors:
            selected = next(iter(source.tensors), None)
        self.explorer.show_tensors(source, selected)
        self.validation.select_source(key)
        self._activate_tensor(selected)

    def _activate_tensor(self, name):
        source = self.sources.get(self.selected_key)
        tensor = source.tensors.get(name) if source else None
        if self.selected_key is not None:
            self.selections[self.selected_key] = name
        if source is not self.inspector.source or tensor is not self.inspector.tensor:
            self.inspector.inspect(source, tensor)

    def show_tensor(self, key, name):
        self.selections[key] = name
        self.selected_key = key
        self._sync_sources()
        self.explorer.search.clear()
        self.explorer.select_tensor(name)
        self._set_page(0)

    def close_file(self):
        if self.selected_key is None:
            return
        self.sources.pop(self.selected_key, None)
        self.selections.pop(self.selected_key, None)
        self.selected_key = next(reversed(self.sources), None)
        self._sync_sources()
        self._message("File closed")

    def open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open tensor files", "", FILE_FILTER)
        if paths:
            self.load_paths(paths)

    def _open_for_combo(self, combo, _checked=False):
        path, _ = QFileDialog.getOpenFileName(self, "Open checkpoint", "", FILE_FILTER)
        if path:
            self.load_paths([path], combo=combo)

    def load_paths(self, paths, *, combo=None):
        candidates = list(dict.fromkeys(str(Path(path).expanduser().resolve()) for path in paths))
        if not candidates:
            return
        existing = dict(self.sources)

        def open_all(progress):
            loaded, errors = [], []
            for index, path in enumerate(candidates):
                progress(index, len(candidates))
                try:
                    loaded.append((path, existing.get(path) or open_file(path)))
                except TensorViewError as error:
                    errors.append(str(error))
            return loaded, errors

        def opened(result):
            loaded, errors = result
            if loaded:
                self.sources.update(loaded)
                self.selected_key = loaded[-1][0]
                self._sync_sources()
                if combo is not None:
                    combo.setCurrentIndex(combo.findData(self.selected_key))
                else:
                    self._set_page(0)
                self._message(f"Opened {len(loaded)} file" + ("s" if len(loaded) != 1 else ""))
            if errors:
                self.show_error("\n\n".join(errors[:6]))

        self.jobs.submit("open", open_all, opened, message="Opening files", replace=False, priority=1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.load_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def _busy(self, message, completed, total):
        self.notice.stop()
        self.status.setText(message or "Ready")
        self.progress.setVisible(bool(message))
        self.progress.setRange(0, 100 if total else 0)
        if total:
            self.progress.setValue(int(100 * completed / total))

    def _message(self, message):
        if self.progress.isVisible():
            return
        self.status.setText(message)
        self.notice.start()

    def show_error(self, details):
        if self.close_pending:
            return
        box = QMessageBox(self)
        box.setWindowTitle("TensorView")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(details[:6000])
        box.exec()

    def _save_to(self, title, name, filters):
        path, selected = QFileDialog.getSaveFileName(self, title, name, filters)
        if not path:
            return None
        raw = Path(path)
        extensions = re.findall(r"\*\.([a-z0-9]+)", filters)
        chosen = re.search(r"\*\.([a-z0-9]+)", selected)
        if raw.suffix.lower().lstrip(".") not in extensions:
            raw = raw.with_name(f"{raw.name}.{chosen.group(1) if chosen else extensions[0]}")
            if raw.exists():
                confirm = QMessageBox.question(self, "Replace file?", f"Replace {raw.name}?")
                if confirm != QMessageBox.StandardButton.Yes:
                    return None
        if str(raw.resolve()) in self.sources:
            self.show_error("Choose an output path different from the files currently open in TensorView.")
            return None
        return raw

    def _convert(self):
        tensor = self.inspector.tensor
        if tensor is None:
            self.show_error("Select a tensor in the explorer before exporting.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Convert or extract tensor")
        dialog.setMinimumWidth(475)
        stack = QVBoxLayout(dialog)
        stack.addWidget(label(tensor.name, "heading", wrap=True))
        stack.addWidget(label("Transpose runs before reshape. Leave fields empty to preserve the current layout.", "caption", wrap=True))
        options = QFormLayout()
        stack.addLayout(options)
        dtype = ComboBox()
        dtype.addItems(["Preserve precision", "float16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool", "complex64", "complex128"])
        if tensor.stored_dtype and tensor.dtype_name != str(tensor.dtype):
            dtype.setItemText(0, f"Decoded {tensor.dtype}")
        axes, shape = QLineEdit(), QLineEdit()
        axes.setPlaceholderText("e.g. 1, 0")
        shape.setPlaceholderText("e.g. 128, -1   or   () for a scalar")
        options.addRow("Output precision", dtype)
        options.addRow("Transpose axes", axes)
        options.addRow("Output shape", shape)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Choose output…")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        stack.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            transpose, reshape = self._integers(axes.text()), self._integers(shape.text())
        except ValueError:
            self.show_error("Enter whole numbers separated by commas, or leave the field empty.")
            return
        filters = ";;".join(f"{name} (*{extension})" for extension, name in TENSOR_EXPORTS.items())
        path = self._save_to("Export tensor", self.inspector.tensor_export_name(), filters)
        if path:
            precision = dtype.currentText() if dtype.currentIndex() else ""
            self.jobs.submit(
                "export", lambda _: export_tensor(tensor, path, dtype=precision, shape=reshape, transpose=transpose),
                lambda _: self._message(f"Saved {path.name}"), message="Exporting tensor", replace=False,
            )

    @staticmethod
    def _integers(text):
        text = text.strip()
        if not text:
            return None
        text = text.removeprefix("(").removesuffix(")").strip()
        return tuple(int(value.strip()) for value in text.rstrip(",").split(",")) if text else ()

    def _save_report(self):
        report = self.pages.currentWidget().report()
        if report is None:
            self.show_error("Finish an inspection, comparison or validation scan before exporting the report.")
            return
        title, info, rows = report
        path = self._save_to("Export report", title.lower().replace(" ", "-"), REPORT_FILTER)
        if path:
            self.jobs.submit(
                "export", lambda _: export_report(path, title, info, rows),
                lambda _: self._message(f"Saved {path.name}"), message="Exporting report", replace=False,
            )

    def _save_plot(self, histogram=False):
        page = self.pages.currentWidget()
        chart = None
        if histogram:
            if self.inspector.stats is not None and self.inspector.stats.sample.size:
                chart = self.inspector.frequency
        elif page is self.inspector:
            chart = self.inspector.active_chart
        elif page is self.comparison and self.comparison.chart_ready:
            chart = self.comparison.export_chart
        if chart is None:
            self.show_error("Select a tensor with a visible plot before exporting a PNG.")
            return
        path = self._save_to("Save plot", "tensorview-plot", "PNG image (*.png)")
        if path:
            try:
                image = chart.png()
                with atomic_destination(path) as destination:
                    if not image.save(str(destination), "PNG"):
                        raise TensorViewError("Qt could not write the plot image.")
                self._message(f"Saved {path.name}")
            except Exception as error:
                self.show_error(str(error))

    def closeEvent(self, event):
        if self.jobs.pool.activeThreadCount():
            self.close_pending = True
            self.jobs.cancel_all()
            self._message("Finishing background work…")
            event.ignore()
            QTimer.singleShot(120, self.close)
        else:
            self.jobs.cancel_all()
            event.accept()
