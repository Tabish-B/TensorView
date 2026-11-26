"""Searchable file and tensor explorer."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .theme import button, label, mono
from .widgets import USER, bytes_text, row, shape_text


class Explorer(QFrame):
    file_changed = Signal(str)
    tensor_changed = Signal(str)
    close_file = Signal()
    open_file = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.source = None
        self.nodes = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 14, 13, 12)
        layout.setSpacing(8)

        self.remove = button("Close", "flat", "Close the selected file.")
        self.remove.clicked.connect(self.close_file)
        layout.addLayout(row(label("FILES", "section"), None, self.remove))
        self.files = QListWidget()
        self.files.setAccessibleName("Open files")
        self.files.setFixedHeight(103)
        self.files.setAlternatingRowColors(False)
        self.files.currentItemChanged.connect(self._file_changed)
        layout.addWidget(self.files)
        separator = QFrame()
        separator.setObjectName("divider")
        layout.addWidget(separator)
        self.caption = label("TENSORS", "section")
        self.count = label("0", "caption")
        layout.addLayout(row(self.caption, None, self.count))

        self.search = QLineEdit()
        self.search.setAccessibleName("Filter tensors")
        self.search.setPlaceholderText("Filter tensors…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setAccessibleName("Tensor explorer")
        self.tree.setHeaderHidden(True)
        self.tree.setFont(mono())
        self.tree.setIndentation(13)
        self.tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(False)
        self.tree.currentItemChanged.connect(self._tensor_changed)
        layout.addWidget(self.tree, 1)
        self.source_summary = label("", "caption", wrap=True)
        layout.addWidget(self.source_summary)
        layout.addWidget(button("Drop files here or browse…", tooltip="Open a tensor or model file."))
        layout.itemAt(layout.count() - 1).widget().clicked.connect(self.open_file)

    def show_files(self, sources, selected):
        self.files.blockSignals(True)
        self.files.clear()
        for key, source in sources.items():
            item = QListWidgetItem(source.label)
            item.setData(USER, key)
            item.setToolTip(f"{source.path or 'Built-in sample'}\n{len(source.tensors):,} tensors")
            self.files.addItem(item)
            if key == selected:
                self.files.setCurrentItem(item)
        self.files.blockSignals(False)
        self.remove.setEnabled(selected is not None)

    def _file_changed(self, current, previous):
        if current is not None:
            self.file_changed.emit(current.data(USER))

    def show_tensors(self, source, selected):
        self.source = source
        self.tree.blockSignals(True)
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()
        self.nodes.clear()
        groups = {}
        if source is not None:
            for name, tensor in sorted(source.tensors.items(), key=lambda pair: pair[0].casefold()):
                parts = name.replace("/", ".").split(".")
                parent = self.tree.invisibleRootItem()
                for depth, part in enumerate(parts[:-1], 1):
                    key = tuple(parts[:depth])
                    item = groups.get(key)
                    if item is None:
                        item = QTreeWidgetItem(parent, [part or "(group)"])
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                        groups[key] = item
                        item.setToolTip(0, ".".join(key))
                    parent = item
                node = QTreeWidgetItem(parent, [parts[-1] or "(unnamed)"])
                node.setData(0, USER, name)
                node.setToolTip(0, f"{name}\n{shape_text(tensor.shape)}  ·  {tensor.dtype_name}")
                self.nodes[name] = node
            self.source_summary.setText(f"{source.size:,} tensor elements  ·  {bytes_text(source.nbytes)}")
            if len(self.nodes) < 300:
                self.tree.expandAll()
            else:
                self.tree.expandToDepth(0)
            if selected is not None:
                self.select_tensor(selected)
        else:
            self.source_summary.setText("Open a local file to explore its tensors.")
        self.tree.setUpdatesEnabled(True)
        self.tree.blockSignals(False)
        self._filter(self.search.text())

    def select_tensor(self, name):
        if name in self.nodes:
            item = self.nodes[name]
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)

    def _tensor_changed(self, current, previous):
        if current is not None:
            name = current.data(0, USER)
            if name is not None:
                self.tensor_changed.emit(name)

    def _filter(self, query):
        term = query.strip().casefold()

        def filter_node(item):
            if item.childCount():
                children = [filter_node(item.child(index)) for index in range(item.childCount())]
                visible = any(children)
                if term and visible:
                    item.setExpanded(True)
            else:
                visible = term in (item.data(0, USER) or "").casefold()
            item.setHidden(not visible)
            return visible

        root = self.tree.invisibleRootItem()
        self.tree.setUpdatesEnabled(False)
        for index in range(root.childCount()):
            filter_node(root.child(index))
        self.tree.setUpdatesEnabled(True)
        visible = sum(not node.isHidden() for node in self.nodes.values())
        self.count.setText(f"{visible:,} / {len(self.nodes):,}" if term else f"{len(self.nodes):,}")
