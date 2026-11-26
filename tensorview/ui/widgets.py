"""Shared UI components for the three workspaces."""

from PySide6.QtCore import QAbstractTableModel, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .charts import compact
from .theme import ACCENT, BLUE, MUTED, YELLOW, label

USER = Qt.ItemDataRole.UserRole
STATUS_COLORS = {
    "Error": "#FFA5B2", "Read error": "#FFA5B2", "Removed": "#FFA5B2", "Review": YELLOW,
    "Added": ACCENT, "Identical": MUTED, "Modified": YELLOW, "Dtype changed": BLUE,
    "Shape changed": BLUE, "Info": BLUE,
}


def shape_text(shape):
    if shape is None:
        return "—"
    return " × ".join(f"{length:,}" for length in shape) if shape else "Scalar"


def bytes_text(size):
    scaled = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if scaled < 1024 or unit == "TiB":
            return f"{size:,} B" if unit == "B" else f"{scaled:.2f} {unit}"
        scaled /= 1024


def panel(margins=(15, 13, 15, 12), spacing=8):
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return frame, layout


def row(*widgets, spacing=8, margins=(0, 0, 0, 0)):
    layout = QHBoxLayout()
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    for widget in widgets:
        if widget is None:
            layout.addStretch()
        else:
            layout.addWidget(widget)
    return layout


def metric(caption):
    frame = QFrame()
    frame.setObjectName("metric")
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    stack = QVBoxLayout(frame)
    stack.setContentsMargins(12, 8, 12, 9)
    stack.setSpacing(3)
    stack.addWidget(label(caption, "caption"))
    value = label("—", "metricValue")
    value.setMinimumWidth(0)
    value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    stack.addWidget(value)
    return frame, value


def wrapped(layout):
    widget = QWidget()
    widget.setLayout(layout)
    return widget


class GridModel(QAbstractTableModel):
    def __init__(self, headers, parent=None):
        super().__init__(parent)
        self.headers, self.rows = headers, []

    def replace(self, rows):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == USER:
            return value
        if role == Qt.ItemDataRole.DisplayRole:
            return value if isinstance(value, str) else compact(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value) if value is not None else None
        if role == Qt.ItemDataRole.ForegroundRole and isinstance(value, str) and value in STATUS_COLORS:
            return QColor(STATUS_COLORS[value])
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


class GridSort(QSortFilterProxyModel):
    def lessThan(self, left, right):
        a, b = left.data(USER), right.data(USER)
        if a is None or b is None:
            return b is None and a is not None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return str(a).casefold() < str(b).casefold()


class Grid(QTableView):
    def __init__(self, headers, *, stretch=0, sort=False, parent=None):
        super().__init__(parent)
        self.grid_model = GridModel(headers, self)
        self.proxy = GridSort(self)
        self.proxy.setSourceModel(self.grid_model)
        self.setModel(self.proxy)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(26)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setMinimumSectionSize(58)
        if stretch is not None:
            self.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(sort)
        if sort:
            self.sortByColumn(0, Qt.SortOrder.AscendingOrder)

    def replace(self, rows):
        self.grid_model.replace(rows)

    def source_row(self):
        index = self.proxy.mapToSource(self.currentIndex())
        return index.row() if index.isValid() else -1
