"""Tensor preview, axis selection, statistics, values, and metadata."""

import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRegularExpression, Qt, QTimer
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis import plot_slice, project, statistics
from ..exporting import tensor_report
from .charts import ACCENT, COLORS, LEVELS, Chart
from .theme import ComboBox, button, label, mono
from .widgets import Grid, bytes_text, metric, panel, row, shape_text, wrapped


class Inspector(QWidget):
    def __init__(self, jobs, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.source = self.tensor = self.stats = None
        self.active_chart = None
        self.axis_edits = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 15)
        layout.setSpacing(12)

        self.file_caption = label("TENSOR INSPECTOR", "section")
        self.title = label("Select a tensor", "title")
        self.title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.quality = label("", "caption")
        heading = QVBoxLayout()
        heading.setSpacing(4)
        heading.addWidget(self.file_caption)
        heading.addWidget(self.title)
        heading.addWidget(self.quality)
        layout.addLayout(heading)

        badges = QHBoxLayout()
        badges.setSpacing(8)
        self.metrics = []
        for name in ("Dimensions", "Data type", "Elements", "Tensor size"):
            frame, content = metric(name)
            self.metrics.append(content)
            badges.addWidget(frame, 1)
        layout.addLayout(badges)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(10)
        layout.addWidget(split, 1)

        chart_card, charts = panel()
        chart_card.setMinimumWidth(365)
        self.plot_title = label("Tensor view", "heading")
        self.mode = ComboBox()
        self.mode.addItems(["Tensor view", "PCA", "t-SNE", "UMAP"])
        self.mode.setToolTip("Choose a tensor slice or a 2D embedding projection.")
        self.reset = button("Reset zoom", "flat")
        self.reset.clicked.connect(self._reset_zoom)
        charts.addLayout(row(self.plot_title, None, self.mode, self.reset))
        self.controls = QWidget()
        self.axes = QHBoxLayout(self.controls)
        self.axes.setContentsMargins(0, 0, 0, 0)
        self.axes.setSpacing(5)
        self.y_axis, self.x_axis = ComboBox(), ComboBox()
        self.y_axis.setAccessibleName("Row axis")
        self.x_axis.setAccessibleName("Column axis")
        self.y_axis.currentIndexChanged.connect(lambda: self._axes_changed(self.y_axis))
        self.x_axis.currentIndexChanged.connect(lambda: self._axes_changed(self.x_axis))
        self.axes.addWidget(label("Y", "caption"))
        self.axes.addWidget(self.y_axis)
        self.axes.addWidget(label("X", "caption"))
        self.axes.addWidget(self.x_axis)
        self.fixed_axes = ComboBox()
        self.fixed_axes.setAccessibleName("Fixed dimension")
        self.fixed_axes.currentIndexChanged.connect(self._fixed_changed)
        self.axes.addWidget(self.fixed_axes)
        self.fixed_value = QLineEdit()
        self.fixed_value.setAccessibleName("Index for fixed dimension")
        self.fixed_value.setMaximumWidth(79)
        self.fixed_value.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]{1,20}"), self))
        self.fixed_value.editingFinished.connect(self._fixed_updated)
        self.axes.addWidget(self.fixed_value)
        self.axes.addStretch()
        charts.addWidget(self.controls)
        self.chart = Chart()
        self.chart.setMinimumHeight(140)
        charts.addWidget(self.chart, 3)
        self.scale_min = label("", "subtle")
        self.scale_max = label("", "subtle")
        self.gradient = QFrame()
        points = ", ".join(f"stop:{level:.3f} {value}" for level, value in zip(LEVELS, COLORS, strict=True))
        self.gradient.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,{points}); border-radius:3px;"
        )
        self.gradient.setFixedSize(95, 6)
        self.scale_key = wrapped(row(self.scale_min, self.gradient, self.scale_max, spacing=5))
        charts.addLayout(row(self.chart.caption, None, self.scale_key))
        separator = QFrame()
        separator.setObjectName("divider")
        charts.addWidget(separator)
        charts.addLayout(row(label("Value distribution", "heading"), None))
        self.frequency = Chart()
        self.frequency.setMinimumHeight(104)
        charts.addWidget(self.frequency, 2)
        charts.addWidget(self.frequency.caption)
        split.addWidget(chart_card)

        info_panel, info = panel(margins=(12, 8, 12, 10))
        info_panel.setMinimumWidth(218)
        info_panel.setMaximumWidth(370)
        tabs = QTabWidget()
        info.addWidget(tabs)
        stats_tab, stats_layout = QWidget(), QVBoxLayout()
        stats_tab.setLayout(stats_layout)
        stats_layout.setContentsMargins(0, 7, 0, 0)
        self.stats_grid = Grid(("Metric", "Value"))
        self.stats_grid.setColumnWidth(1, 112)
        stats_layout.addWidget(self.stats_grid)
        self.stats_note = label("", "caption", wrap=True)
        stats_layout.addWidget(self.stats_note)
        tabs.addTab(stats_tab, "Statistics")
        values_tab, values_layout = QWidget(), QVBoxLayout()
        values_tab.setLayout(values_layout)
        values_layout.setContentsMargins(0, 7, 0, 0)
        self.values = Grid(("Flat index", "Value"), stretch=1)
        self.values.setColumnWidth(0, 72)
        values_layout.addWidget(self.values)
        self.values_note = label("", "caption", wrap=True)
        values_layout.addWidget(self.values_note)
        self.lookup = QLineEdit()
        self.lookup.setPlaceholderText("Look up a flat index…")
        self.lookup.setAccessibleName("Tensor value index")
        self.lookup.setToolTip("Use a zero-based flat index, or comma-separated coordinates.")
        self.lookup.returnPressed.connect(self._lookup_value)
        values_layout.addWidget(self.lookup)
        self.lookup_result = label("", "caption", wrap=True)
        self.lookup_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        values_layout.addWidget(self.lookup_result)
        tabs.addTab(values_tab, "Values")
        self.metadata = QPlainTextEdit()
        self.metadata.setFont(mono())
        self.metadata.setReadOnly(True)
        self.metadata.setPlaceholderText("Open a file to see metadata.")
        tabs.addTab(self.metadata, "File")
        split.addWidget(info_panel)
        split.setStretchFactor(0, 1)
        split.setSizes([640, 282])
        self.scale_key.hide()
        self.controls.hide()
        self.mode.currentIndexChanged.connect(self.refresh_plot)
        self.slice_timer = QTimer(self)
        self.slice_timer.setSingleShot(True)
        self.slice_timer.setInterval(120)
        self.slice_timer.timeout.connect(self.refresh_plot)

    def inspect(self, source, tensor):
        for group in ("tensor-statistics", "tensor-preview", "tensor-slice", "tensor-lookup"):
            self.jobs.cancel(group)
        self.source, self.tensor, self.stats = source, tensor, None
        self.active_chart = None
        self.slice_timer.stop()
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.setEnabled(tensor is not None and len(tensor.shape) == 2 and min(tensor.shape) >= 2)
        self.mode.blockSignals(False)
        self.scale_key.hide()
        self.stats_grid.replace([])
        self.values.replace([])
        self.values_note.setText("")
        self.lookup.clear()
        self.lookup.setEnabled(tensor is not None and tensor.size > 0)
        self.lookup_result.setText("")
        self.stats_note.setText("Calculating statistics…" if tensor else "")
        self.frequency.create_placeholder("Calculating distribution…" if tensor else "Select a tensor.")
        self.file_caption.setText(f"{source.label}  /  {source.format}" if source else "TENSOR INSPECTOR")
        self._metadata(source)
        if tensor is None:
            self.controls.hide()
            self.title.setText("No tensor selected")
            self.quality.setText("Open a file or choose a tensor from the explorer.")
            self.chart.create_placeholder("Select a tensor to see its values.")
            for value in self.metrics:
                value.setText("—")
            return
        self.title.setText(tensor.name)
        self.quality.setText("Scanning tensor…")
        for widget, value in zip(
            self.metrics, (shape_text(tensor.shape), tensor.dtype_name, f"{tensor.size:,}", bytes_text(tensor.nbytes)),
            strict=True,
        ):
            widget.setText(value)
            widget.setToolTip(value)
        self._setup_axes()
        self.refresh_plot()
        self.jobs.submit(
            "tensor-preview", lambda _: tensor.preview(2048), self._preview,
            error_callback=lambda error: self.values_note.setText(error), priority=1,
        )
        self.jobs.submit(
            "tensor-statistics", lambda progress: statistics(tensor, progress), self._statistics,
            message=f"Scanning {tensor.name}", error_callback=self._statistics_error,
        )

    def _metadata(self, source):
        if source is None:
            self.metadata.clear()
            return
        lines = [
            source.label, source.format, str(source.path or "Built-in sample"),
            "", f"{len(source.tensors):,} tensors", f"{source.size:,} elements", bytes_text(source.nbytes),
        ]
        if source.metadata:
            lines.extend(["", "Metadata", *(f"{key} = {value}" for key, value in source.metadata.items())])
        if source.notes:
            lines.extend(["", "Reader notes", *source.notes])
        self.metadata.setPlainText("\n".join(lines))

    def _setup_axes(self):
        ndim = len(self.tensor.shape)
        self.axis_edits.clear()
        for combo, position in ((self.y_axis, ndim - 2), (self.x_axis, ndim - 1)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(axis) for axis in range(ndim)])
            combo.setCurrentIndex(position)
            combo.blockSignals(False)
        self._refresh_fixed()

    def _axes_changed(self, changed):
        if self.x_axis.currentIndex() == self.y_axis.currentIndex():
            other = self.y_axis if changed is self.x_axis else self.x_axis
            other.blockSignals(True)
            other.setCurrentIndex((changed.currentIndex() + 1) % other.count())
            other.blockSignals(False)
        self._refresh_fixed()
        self.slice_timer.start()

    def _refresh_fixed(self):
        dims = self.tensor.shape
        previous = self.fixed_axes.currentData()
        excluded = (self.y_axis.currentIndex(), self.x_axis.currentIndex())
        self.fixed_axes.blockSignals(True)
        self.fixed_axes.clear()
        for axis in range(len(dims)):
            if axis not in excluded:
                self.fixed_axes.addItem(f"Axis {axis} index", axis)
                self.axis_edits.setdefault(axis, 0)
        prior = self.fixed_axes.findData(previous)
        self.fixed_axes.setCurrentIndex(max(0, prior))
        self.fixed_axes.blockSignals(False)
        self.fixed_axes.setVisible(len(dims) > 2)
        self.fixed_value.setVisible(len(dims) > 2)
        self._fixed_changed()

    def _fixed_changed(self):
        axis = self.fixed_axes.currentData()
        if axis is not None:
            self.fixed_value.setText(str(self.axis_edits.get(axis, 0)))
            maximum = max(0, self.tensor.shape[axis] - 1)
            self.fixed_value.setToolTip(f"Index from 0 to {maximum:,}")

    def _fixed_updated(self):
        axis = self.fixed_axes.currentData()
        if axis is None or self.tensor is None:
            return
        index = int(self.fixed_value.text() or "0")
        bounded = min(index, max(0, self.tensor.shape[axis] - 1))
        self.fixed_value.setText(str(bounded))
        if bounded != self.axis_edits[axis]:
            self.axis_edits[axis] = bounded
            self.slice_timer.start()

    def refresh_plot(self):
        tensor = self.tensor
        if tensor is None:
            return
        self.scale_key.hide()
        self.active_chart = None
        method = self.mode.currentText()
        self.controls.setVisible(method == "Tensor view" and len(tensor.shape) >= 2)
        if method != "Tensor view":
            self.plot_title.setText(f"{method} projection")
            self.chart.create_placeholder(f"Calculating {method}…")
            self.jobs.submit(
                "tensor-slice", lambda _: project(tensor, method),
                lambda results: self._projection(results, method),
                message=f"Calculating {method}", error_callback=self._plot_error, priority=1,
            )
            return
        dims = len(tensor.shape)
        axes = (self.y_axis.currentIndex(), self.x_axis.currentIndex()) if dims > 1 else tuple(range(dims))
        fixed = dict(self.axis_edits)
        view_name = ("Tensor value" if dims == 0 else "Signal preview" if dims == 1 else "Tensor heatmap")
        self.plot_title.setText(view_name + (" · magnitude" if tensor.dtype.kind == "c" else ""))
        self.chart.create_placeholder("Drawing tensor…")
        self.jobs.submit(
            "tensor-slice", lambda _: plot_slice(tensor, axes, fixed), lambda result: self._draw(result, axes),
            error_callback=self._plot_error, priority=1,
        )

    def _plot_error(self, message):
        self.chart.create_placeholder("This view could not be drawn.")
        self.chart.caption.setText(message)
        self.chart.caption.setToolTip(message)

    def _draw(self, result, axes):
        bounds = self.chart.show_slice(*result, axes)
        self.active_chart = self.chart if result[0].size and np.any(np.isfinite(result[0])) else None
        if bounds is not None:
            self.scale_min.setText(bounds[0])
            self.scale_max.setText(bounds[1])
            self.scale_key.show()

    def _projection(self, results, method):
        self.chart.projection(*results, method)
        self.active_chart = self.chart

    def _reset_zoom(self):
        if self.active_chart is not None:
            self.active_chart.autoRange()
        if self.stats is not None and self.stats.sample.size:
            self.frequency.autoRange()

    def _preview(self, values):
        self.values.replace(
            (index, str(value.item())) for index, value in enumerate(values)
        )
        self.values_note.setText(f"Showing {values.size:,} of {self.tensor.size:,} values in row-major order.")

    def _lookup_value(self):
        tensor = self.tensor
        text = self.lookup.text().strip()
        if tensor is None or not text:
            return
        try:
            indices = tuple(int(piece.strip()) for piece in text.split(","))
            if len(indices) == 1:
                index = indices[0]
                if not 0 <= index < tensor.size:
                    raise ValueError
                coordinates = []
                for length in reversed(tensor.shape):
                    index, coordinate = divmod(index, length)
                    coordinates.append(coordinate)
                indices = tuple(reversed(coordinates))
            elif len(indices) != len(tensor.shape) or any(
                not 0 <= index < length for index, length in zip(indices, tensor.shape, strict=True)
            ):
                raise ValueError
        except ValueError:
            self.jobs.cancel("tensor-lookup")
            self.lookup_result.setText("Enter a valid flat index or one coordinate per axis.")
            return

        self.jobs.submit(
            "tensor-lookup", lambda _: tensor.read(indices).item(),
            lambda value: self.lookup_result.setText(f"{indices} = {value}"),
            error_callback=self.lookup_result.setText, priority=1,
        )

    def _statistics_error(self, error):
        self.stats_note.setText(error)
        self.quality.setText("Tensor scan failed")
        self.frequency.create_placeholder("Distribution unavailable.")

    def _statistics(self, stats):
        self.stats = stats
        self.stats_grid.replace([
            ("Minimum", stats.minimum), ("Maximum", stats.maximum), ("Mean", stats.mean),
            ("Median", stats.median), ("Std. deviation", stats.std), ("Sparsity", f"{stats.sparsity:.2%}"),
            ("Finite values", stats.finite), ("NaN", stats.nan), ("Infinite", stats.infinite),
            ("Zero values", stats.zeros), *((f"P{k}", value) for k, value in stats.percentiles.items()),
        ])
        notes = []
        if stats.complex:
            notes.append("Plots and summary values use magnitude.")
        if stats.sampled:
            notes.append("Percentiles and distribution use a sample.")
        if not stats.finite and stats.count:
            notes.append("No finite values.")
        self.stats_note.setText(" ".join(notes))
        if stats.nan or stats.infinite:
            self.quality.setText(f"{stats.nan:,} NaN  ·  {stats.infinite:,} infinite")
        elif stats.count:
            self.quality.setText(f"{stats.finite:,} finite values")
        else:
            self.quality.setText("Empty tensor")
        self.frequency.histogram([("Sample" if stats.sampled else "Values", stats.sample, ACCENT)])

    def report(self):
        if self.tensor is not None and self.stats is not None:
            return tensor_report(self.source, self.tensor, self.stats)

    def tensor_export_name(self):
        if self.tensor is None:
            return "tensor"
        return re.sub(r"[^\w-]+", "-", Path(self.tensor.name).name).strip("-")[:95] or "tensor"
