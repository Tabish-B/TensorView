"""Checkpoint comparison and numerical validation workspaces."""

from collections import Counter
from functools import partial

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QStackedWidget, QVBoxLayout, QWidget

from ..analysis import compare, plot_slice, statistics, validate
from ..exporting import comparison_report, validation_report
from .charts import BASELINE_COLOR, CANDIDATE_COLOR, Chart, compact
from .theme import ComboBox, button, label
from .widgets import Grid, metric, panel, row, shape_text, wrapped


def source_picker(combo, sources):
    selected = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for key, source in sources.items():
        combo.addItem(source.label, key)
        if source.path:
            combo.setItemData(combo.count() - 1, str(source.path), Qt.ItemDataRole.ToolTipRole)
    previous = combo.findData(selected)
    if previous >= 0:
        combo.setCurrentIndex(previous)
    combo.blockSignals(False)


class Workspace(QWidget):
    def __init__(self, title, description):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 18, 20, 16)
        self.layout.setSpacing(12)
        self.layout.addWidget(label(title, "title"))
        self.layout.addWidget(label(description, "muted", wrap=True))
        self.sources = {}

    def _metrics(self, names):
        frames, values = [], []
        for name in names:
            card, value = metric(name)
            frames.append(card)
            values.append(value)
        self.layout.addLayout(row(*frames))
        return values


class Comparison(Workspace):
    def __init__(self, jobs, open_one):
        super().__init__("Checkpoint comparison", "Match tensors by name and measure changes from baseline to candidate.")
        self.jobs = jobs
        self.result = None
        self.chart_ready = False

        pickers, boxes = panel()
        self.baseline, self.candidate = ComboBox(), ComboBox()
        for description, combo in (("Baseline", self.baseline), ("Candidate", self.candidate)):
            title = label(description, "section")
            combo.setAccessibleName(description)
            combo.setMinimumContentsLength(11)
            combo.currentIndexChanged.connect(self.invalidate)
            open_button = button("Browse…", "flat", f"Open a file as {description.lower()}.")
            open_button.clicked.connect(partial(open_one, combo))
            boxes.addLayout(row(title, combo, open_button, None))
            boxes.itemAt(boxes.count() - 1).setStretch(1, 2)
            boxes.itemAt(boxes.count() - 1).setStretch(3, 1)
        self.run = button("Run comparison", "primary")
        self.run.clicked.connect(self.start)
        self.message = label("Choose two files, then run a comparison.", "caption", wrap=True)
        actions = row(self.message, self.run)
        actions.setStretch(0, 1)
        boxes.addLayout(actions)
        self.layout.addWidget(pickers)
        self.metrics = self._metrics(("Baseline elements", "Candidate elements", "Modified", "Added / Removed"))

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(10)
        table_panel, table_layout = panel()
        self.table = Grid(
            ("Tensor", "Status", "A shape", "B shape", "Changed", "Relative L2", "Cosine"),
            sort=True,
        )
        for column, width in enumerate((0, 123, 91, 91, 87, 102, 78)):
            if column:
                self.table.setColumnWidth(column, width)
        self.table.selectionModel().currentRowChanged.connect(self._select)
        table_layout.addWidget(self.table)
        split.addWidget(table_panel)

        chart_panel, chart_layout = panel()
        self.detail = label("Tensor distribution", "heading")
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary = label("Select a tensor in the results to compare its values.", "caption")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.chart_mode = ComboBox()
        self.chart_mode.addItems(["Distribution", "Side by side"])
        self.chart_mode.setAccessibleName("Comparison plot view")
        self.chart_mode.currentIndexChanged.connect(self._select)
        self.chart = Chart()
        self.chart.create_placeholder("Select a comparison result.")
        self.views = QStackedWidget()
        distribution = QVBoxLayout()
        distribution.setContentsMargins(0, 0, 0, 0)
        distribution.addWidget(self.chart, 1)
        distribution.addWidget(self.chart.caption)
        self.views.addWidget(wrapped(distribution))
        self.side_charts = []
        side_views = QSplitter(Qt.Orientation.Horizontal)
        side_views.setHandleWidth(9)
        side_views.setChildrenCollapsible(False)
        for name in ("Baseline", "Candidate"):
            chart = Chart()
            chart.create_placeholder(f"No {name.lower()} tensor selected.")
            self.side_charts.append(chart)
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(label(name, "caption"))
            column.addWidget(chart, 1)
            column.addWidget(chart.caption)
            side_views.addWidget(wrapped(column))
        self.views.addWidget(side_views)
        self.export_chart = None
        chart_layout.addLayout(row(self.detail, None, self.chart_mode))
        chart_layout.addWidget(self.summary)
        chart_layout.addWidget(self.views, 1)
        split.addWidget(chart_panel)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 170])
        self.layout.addWidget(split, 1)
        self.invalidate()

    def set_sources(self, sources):
        self.sources = sources
        old = self.baseline.currentData(), self.candidate.currentData()
        source_picker(self.baseline, sources)
        source_picker(self.candidate, sources)
        if old[1] is None and self.candidate.count() > 1:
            self.candidate.blockSignals(True)
            self.candidate.setCurrentIndex(1)
            self.candidate.blockSignals(False)
        if old != (self.baseline.currentData(), self.candidate.currentData()):
            self.invalidate()

    def invalidate(self):
        self.jobs.cancel("comparison")
        self.jobs.cancel("comparison-charts")
        self.result = None
        self.table.replace([])
        self._select()
        a, b = self.baseline.currentData(), self.candidate.currentData()
        self.run.setEnabled(a is not None and b is not None and a != b)
        self.message.setText("Choose two different files, then run a comparison.")
        for value in self.metrics:
            value.setText("—")

    def start(self):
        a = self.sources.get(self.baseline.currentData())
        b = self.sources.get(self.candidate.currentData())
        if not a or not b or a is b:
            return
        self.invalidate()
        self.run.setEnabled(False)
        self.message.setText("Comparing named tensors…")
        self.jobs.submit(
            "comparison", lambda progress: compare(a, b, progress),
            lambda items: self._populate(a, b, items), message="Comparing checkpoints", error_callback=self._error,
        )

    def _error(self, message):
        self.run.setEnabled(True)
        self.message.setText(message)

    def _populate(self, a, b, items):
        self.result = a, b, items
        self.table.replace(
            (d.name, d.status, shape_text(d.shape_a), shape_text(d.shape_b),
             d.changed_values, d.relative_l2, d.cosine)
            for d in items
        )
        counts = Counter(d.status for d in items)
        details = f"{len(items):,} tensors  ·  {counts['Identical']:,} identical"
        for state in ("Shape changed", "Dtype changed", "Read error"):
            if counts[state]:
                details += f"  ·  {counts[state]:,} {state.lower()}"
        self.message.setText(details)
        values = (f"{a.size:,}", f"{b.size:,}", f"{counts['Modified']:,}", f"{counts['Added']:,} / {counts['Removed']:,}")
        for field, value in zip(self.metrics, values, strict=True):
            field.setText(value)
            field.setToolTip(value)
        self.run.setEnabled(True)
        if items:
            self.table.selectRow(0)

    def _select(self, *_):
        self.jobs.cancel("comparison-charts")
        self.chart_ready = False
        self.export_chart = None
        side_by_side = self.chart_mode.currentIndex() == 1
        self.views.setCurrentIndex(int(side_by_side))
        index = self.table.source_row()
        if index < 0 or self.result is None:
            self.detail.setText("Tensor distribution")
            self.summary.setText("Select a tensor in the results to compare its values.")
            self.chart.create_placeholder("Select a comparison result.")
            for chart in self.side_charts:
                chart.create_placeholder("Select a comparison result.")
            return
        a, b, items = self.result
        difference = items[index]
        self.detail.setText(difference.name)
        self.detail.setToolTip(difference.name)
        self.summary.setText(
            f"Baseline {difference.dtype_a or '—'} · {difference.count_a:,} elements    "
            f"Candidate {difference.dtype_b or '—'} · {difference.count_b:,} elements\n"
            f"Mean |Δ| {compact(difference.mean_absolute)}    "
            f"Max |Δ| {compact(difference.max_absolute)}    Finite pairs {compact(difference.finite_pairs)}"
            + (f"\n{difference.detail}" if difference.detail else "")
        )
        if side_by_side:
            self._request_slices(a, b, difference.name)
            return
        self.chart.create_placeholder("Calculating distributions…")

        def distributions(progress):
            series = []
            for title, source, color in (("Baseline", a, BASELINE_COLOR), ("Candidate", b, CANDIDATE_COLOR)):
                tensor = source.tensors.get(difference.name)
                if tensor is not None:
                    stats = statistics(tensor, progress)
                    if stats.sampled:
                        title += " sample"
                    if stats.complex:
                        title += " magnitude"
                    series.append((title, stats.sample, color))
            return series

        self.jobs.submit(
            "comparison-charts", distributions, self._chart_data, priority=1, error_callback=self._chart_error,
        )

    def _chart_data(self, series):
        self.chart.histogram(series)
        self.chart_ready = any(values.size for _, values, _ in series)
        self.export_chart = self.chart if self.chart_ready else None

    def _request_slices(self, a, b, name):
        for chart in self.side_charts:
            chart.create_placeholder("Drawing tensor slice…")

        def slices(_):
            result = []
            for source in (a, b):
                tensor = source.tensors.get(name)
                if tensor is None:
                    result.append(None)
                    continue
                dims = len(tensor.shape)
                axes = tuple(range(max(0, dims - 2), dims))
                result.append((*plot_slice(tensor, axes, {}), axes, tensor.dtype.kind == "c"))
            return result

        self.jobs.submit(
            "comparison-charts", slices, self._side_data, priority=1, error_callback=self._chart_error,
        )

    def _side_data(self, series):
        import numpy as np

        finite_values = [values[np.isfinite(values)] for values, *_ in filter(None, series)]
        finite_values = [values for values in finite_values if values.size]
        scale = max((float(np.max(np.abs(values))) for values in finite_values), default=0) or 1
        combined = np.concatenate([values / scale for values in finite_values]) if finite_values else np.empty(0)
        bounds = tuple(np.percentile(combined, [1, 99])) if combined.size else None
        if bounds is not None and bounds[0] == bounds[1]:
            bounds = None
        finite_ranges = None if not finite_values else (
            min(float(values.min()) for values in finite_values),
            max(float(values.max()) for values in finite_values),
        )
        for chart, data in zip(self.side_charts, series, strict=True):
            if data is None:
                chart.create_placeholder("Tensor is absent in this checkpoint.")
            else:
                values, steps, lengths, axes, magnitude = data
                chart.show_slice(values, steps, lengths, axes, scale_bounds=(scale, bounds))
                if values.size and np.any(np.isfinite(values)):
                    if values.ndim < 2 and finite_ranges[0] < finite_ranges[1]:
                        chart.setYRange(*finite_ranges, padding=0.08)
                    shared_range = "" if bounds is None or values.ndim < 2 else " · Shared color scale"
                    chart.caption.setText(chart.caption.text() + (" · Magnitude" if magnitude else "") + shared_range)
                    self.chart_ready = True
                    self.export_chart = self.export_chart or chart

    def _chart_error(self, message):
        self.chart.create_placeholder("Distribution unavailable.")
        self.chart.caption.setText(message)
        for chart in self.side_charts:
            chart.create_placeholder("Tensor slice unavailable.")
            chart.caption.setText(message)

    def report(self):
        return comparison_report(*self.result) if self.result is not None else None


class Validation(Workspace):
    def __init__(self, jobs, show_tensor):
        super().__init__("Numerical validation", "Scan every tensor for non-finite values and conditions worth reviewing.")
        self.jobs, self.show_tensor = jobs, show_tensor
        self.result = None
        pickers, layout = panel()
        self.source = ComboBox()
        self.source.setAccessibleName("File to validate")
        self.source.currentIndexChanged.connect(self.invalidate)
        self.run = button("Run validation", "primary")
        self.run.clicked.connect(self.start)
        selectors = row(label("File", "section"), self.source, None, self.run)
        selectors.setStretch(1, 2)
        selectors.setStretch(2, 1)
        layout.addLayout(selectors)
        self.message = label("", "caption", wrap=True)
        layout.addWidget(self.message)
        self.layout.addWidget(pickers)
        self.metrics = self._metrics(("Tensors scanned", "Errors", "Review", "Information"))
        card, results = panel()
        self.table = Grid(("Level", "Tensor", "Check", "Details"), stretch=3, sort=True)
        self.table.setColumnWidth(0, 78)
        self.table.setColumnWidth(1, 205)
        self.table.setColumnWidth(2, 170)
        self.table.doubleClicked.connect(self._inspect_row)
        results.addWidget(self.table, 1)
        self.layout.addWidget(card, 1)
        self.invalidate()

    def set_sources(self, sources):
        previous = self.source.currentData()
        self.sources = sources
        source_picker(self.source, sources)
        if previous != self.source.currentData():
            self.invalidate()

    def select_source(self, key):
        index = self.source.findData(key)
        if index >= 0:
            self.source.setCurrentIndex(index)

    def invalidate(self):
        self.jobs.cancel("validation")
        self.result = None
        self.table.replace([])
        self.run.setEnabled(self.source.count() > 0)
        self.message.setText("Double-click a finding to open its tensor in the inspector.")
        for value in self.metrics:
            value.setText("—")

    def start(self):
        key = self.source.currentData()
        source = self.sources.get(key)
        if source is None:
            return
        self.invalidate()
        self.run.setEnabled(False)
        self.message.setText("Scanning every tensor…")
        self.jobs.submit(
            "validation", lambda progress: validate(source, progress),
            lambda findings: self._populate(key, source, findings), message=f"Validating {source.label}",
            error_callback=self._error,
        )

    def _error(self, message):
        self.run.setEnabled(True)
        self.message.setText(message)

    def _populate(self, key, source, findings):
        self.result = key, source, findings
        totals = Counter(item.level for item in findings)
        self.table.replace((item.level, item.tensor, item.check, item.detail) for item in findings)
        for field, value in zip(
            self.metrics, (len(source.tensors), totals["Error"], totals["Review"], totals["Info"]), strict=True,
        ):
            field.setText(f"{value:,}")
        message = "No numerical findings." if not findings else "Double-click a finding to inspect its tensor."
        self.message.setText(message)
        self.run.setEnabled(True)

    def _inspect_row(self, _):
        index = self.table.source_row()
        if index >= 0 and self.result is not None:
            key, source, findings = self.result
            name = findings[index].tensor
            if name in source.tensors:
                self.show_tensor(key, name)

    def report(self):
        if self.result is not None:
            _, source, findings = self.result
            return validation_report(source, findings)
