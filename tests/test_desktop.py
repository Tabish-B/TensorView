import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def wait_for(app, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.015)
    pytest.fail("The desktop workflow did not finish.")


def test_inspect_compare_validate_and_open_file(tmp_path):
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent
    from PySide6.QtWidgets import QApplication

    from tensorview.ui.theme import apply_theme
    from tensorview.ui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    window = MainWindow()
    window.show()
    screenshots = os.environ.get("TENSORVIEW_TEST_SCREENSHOTS")

    def capture(name):
        if screenshots:
            app.processEvents()
            assert window.grab().save(str(Path(screenshots) / f"tensorview-{name}.png"))

    try:
        inspector = window.inspector
        wait_for(app, lambda: inspector.stats is not None and inspector.active_chart is not None)
        assert inspector.stats.count == 15_360
        assert inspector.report() is not None
        capture("inspector")
        assert inspector.chart.png().width() == 2400

        window.show_tensor("example:baseline", "features.layer_stack")
        wait_for(app, lambda: inspector.active_chart is not None)
        inspector.fixed_value.setText("2")
        inspector._fixed_updated()
        wait_for(app, lambda: inspector.chart.image_values is not None and inspector.chart.image_values[0][0, 0]
                 == inspector.tensor.read((2, 0, 0)))
        window.explorer.search.setText("signal")
        assert window.explorer.count.text().startswith("1 /")
        window.explorer.search.clear()
        inspector.lookup.setText("2, 0, 0")
        inspector._lookup_value()
        wait_for(app, lambda: inspector.lookup_result.text().startswith("(2, 0, 0) = "))

        window.navigation.button(1).click()
        window.comparison.run.click()
        wait_for(app, lambda: window.comparison.result is not None and window.comparison.chart_ready)
        assert len(window.comparison.result[2]) == 7
        assert window.comparison.report() is not None
        capture("comparison")
        comparison = window.comparison
        for row in range(comparison.table.proxy.rowCount()):
            if comparison.table.proxy.index(row, 0).data() == "encoder.projection.weight":
                comparison.table.selectRow(row)
                break
        comparison.chart_mode.setCurrentIndex(1)
        wait_for(app, lambda: all(chart.image_values is not None for chart in comparison.side_charts))
        capture("side-by-side")

        window.navigation.button(2).click()
        window.validation.run.click()
        wait_for(app, lambda: window.validation.result is not None)
        assert window.validation.result[2][0].check == "All zero"
        capture("validation")

        path = tmp_path / "incoming.npy"
        np.save(path, [float("nan"), 5, 8, 12])
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        event = QDropEvent(
            QPointF(100, 100), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        window.dropEvent(event)
        wait_for(app, lambda: inspector.stats is not None and inspector.source.path == path.resolve())
        assert inspector.stats.nan == 1
        assert window.pages.currentWidget() is inspector
    finally:
        window.close()
        wait_for(app, lambda: not window.isVisible())
