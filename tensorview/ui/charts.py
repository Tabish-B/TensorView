"""Interactive PyQtGraph charts for tensor slices and distributions."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage

from .theme import ACCENT, BLUE, BORDER, MUTED, PLOT, YELLOW, label

COLORS = ("#21224F", "#324D8A", "#2F789B", "#49C6BA", "#D8EDD1", "#FFCA92")
LEVELS = (0.0, 0.22, 0.42, 0.60, 0.79, 1.0)


def compact(value):
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.5g}"


class Chart(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent, background=PLOT)
        self.setMinimumHeight(110)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=0.16)
        self.setMouseEnabled(x=True, y=True)
        for side in ("left", "bottom"):
            axis = self.getAxis(side)
            axis.setPen(pg.mkPen(BORDER))
            axis.setTextPen(pg.mkPen(MUTED))
        self.caption = label("", "caption")
        self.caption.setMinimumWidth(0)
        self.image_values = None
        self.points = None
        self.motion = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=22, slot=self._hover)
        self.create_placeholder("Select a tensor to see its values.")

    def _prepare(self, x="", y=""):
        self.clear()
        self.image_values = None
        self.points = None
        self.setLabel("bottom", x, color=MUTED)
        self.setLabel("left", y, color=MUTED)
        self.setAspectLocked(False)
        self.getViewBox().invertY(False)
        self.enableAutoRange()
        self.caption.setText("")
        self.caption.setToolTip("")

    def create_placeholder(self, message):
        self._prepare()
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.enableAutoRange(False)
        self.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        item = pg.TextItem(text=message, color=MUTED, anchor=(0.5, 0.5))
        item.setPos(0.5, 0.5)
        self.addItem(item, ignoreBounds=True)

    def show_slice(self, values, steps, lengths, axes, scale_bounds=None):
        if values.size == 0:
            self.create_placeholder("This tensor is empty.")
            return None
        finite = values[np.isfinite(values)]
        if not finite.size:
            self.create_placeholder("This view contains only NaN or infinity.")
            return None
        self.showAxis("bottom")
        self.showAxis("left")
        if values.ndim < 2:
            self._prepare("Index", "Value")
            values = values.reshape(-1)
            x = np.arange(values.size) * (steps[0] if steps else 1)
            self.plot(
                x, np.where(np.isfinite(values), values, np.nan), pen=pg.mkPen(ACCENT, width=1.6),
                symbol="o" if values.size <= 30 else None, symbolSize=5, symbolBrush=ACCENT, connect="finite",
            )
            self.caption.setText(f"{values.size:,} plotted values")
            return None
        self._prepare(f"Axis {axes[1]}", f"Axis {axes[0]}")
        self.getViewBox().invertY(True)
        self.image_values = (values, steps, lengths)
        scale, bounds = scale_bounds or (float(np.max(np.abs(finite))) or 1.0, None)
        scaled = finite / scale
        low, high = bounds or tuple(float(x) for x in np.percentile(scaled, [1, 99]))
        if low == high:
            low, high = (float(scaled.min()), float(scaled.max()))
        if low == high:
            low, high = low - 0.5, high + 0.5
        image = pg.ImageItem(axisOrder="row-major")
        colors = [QColor(value) for value in COLORS]
        palette = pg.ColorMap(LEVELS, colors).getLookupTable(nPts=256, alpha=True)
        image.setLookupTable(palette)
        display = np.where(np.isfinite(values), values / scale, np.nan).astype(np.float32)
        image.setImage(display, autoLevels=False, levels=(low, high))
        image.setRect(QRectF(-0.5, -0.5, values.shape[1] * steps[1], values.shape[0] * steps[0]))
        self.addItem(image)
        self.setRange(xRange=(-0.5, lengths[1] - 0.5), yRange=(-0.5, lengths[0] - 0.5), padding=0)
        self.caption.setText(f"{values.shape[0]:,} × {values.shape[1]:,} plotted values")
        return compact(low * scale), compact(high * scale)

    def histogram(self, series):
        present = [(name, data, color) for name, data, color in series if data.size]
        if not present:
            self.create_placeholder("No finite values to plot.")
            return
        self._prepare("Value", "Share")
        self.showAxis("bottom")
        self.showAxis("left")
        scale = max(float(np.max(np.abs(data))) for _, data, _ in present) or 1.0
        # Normalizing the histogram edges avoids overflow at finite float64 extremes.
        axis_label = "Value"
        if scale > 1e100 or scale < 1e-100:
            axis_label = f"Value / {compact(scale)}"
        else:
            scale = 1.0
        self.setLabel("bottom", axis_label, color=MUTED)
        lower = min(float(data.min() / scale) for _, data, _ in present)
        upper = max(float(data.max() / scale) for _, data, _ in present)
        if lower == upper:
            padding = abs(lower) * 0.05 or 0.5
            lower, upper = lower - padding, upper + padding
        bins = np.linspace(lower, upper, 49)
        for _, data, color in present:
            counts, _ = np.histogram(data / scale, bins=bins)
            brush = QColor(color)
            brush.setAlpha(60)
            self.plot(
                bins, counts / data.size, stepMode="center",
                pen=pg.mkPen(color, width=1.45), fillLevel=0, brush=brush,
            )
        self.caption.setText("  ·  ".join(f"{name} {data.size:,}" for name, data, _ in present))

    def projection(self, result, indices, method):
        self._prepare(f"{method} 1", f"{method} 2")
        self.showAxis("bottom")
        self.showAxis("left")
        self.plot(
            result[:, 0], result[:, 1], pen=None, symbol="o", symbolSize=6, symbolPen=None,
            symbolBrush=pg.mkBrush(85, 222, 197, 150),
        )
        self.points = (result, indices)
        self.caption.setText(f"{len(indices):,} embedded rows")

    def _hover(self, event):
        point = event[0]
        if not self.plotItem.vb.sceneBoundingRect().contains(point):
            return
        point = self.plotItem.vb.mapSceneToView(point)
        if self.image_values is not None:
            values, steps, lengths = self.image_values
            row, column = int((point.y() + 0.5) // steps[0]), int((point.x() + 0.5) // steps[1])
            if 0 <= row < values.shape[0] and 0 <= column < values.shape[1]:
                self.caption.setText(
                    f"Row {row * steps[0]:,}   Column {column * steps[1]:,}   Value {compact(values[row, column])}"
                )
            else:
                self.caption.setText(f"Slice dimensions {lengths[0]:,} × {lengths[1]:,}")
        elif self.points is not None:
            coordinates, indices = self.points
            vrange = self.viewRange()
            scale = [max(high - low, 1e-10) for low, high in vrange]
            squared = np.square((coordinates - [point.x(), point.y()]) / scale).sum(axis=1)
            nearest = int(squared.argmin())
            if squared[nearest] < 0.001:
                self.caption.setText(f"Embedding row {indices[nearest]:,}")

    def png(self) -> QImage:
        exporter = ImageExporter(self.plotItem)
        exporter.parameters()["width"] = 2400
        return exporter.export(toBytes=True)


BASELINE_COLOR, CANDIDATE_COLOR = BLUE, YELLOW
