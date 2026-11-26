"""Desktop entry point and operating-system file-open events."""

import argparse
import sys

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from . import __version__


class TensorApplication(QApplication):
    window = None
    system_paths = None

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen and event.file():
            if self.window is None:
                if self.system_paths is None:
                    self.system_paths = []
                self.system_paths.append(event.file())
            else:
                self.window.load_paths([event.file()])
            return True
        return super().event(event)


def main():
    parser = argparse.ArgumentParser(description="Open a local TensorView desktop session.")
    parser.add_argument("files", nargs="*", help="Tensor or model files to open")
    parser.add_argument("--empty", action="store_true", help="Start without built-in example checkpoints")
    parser.add_argument("--version", action="version", version=f"TensorView {__version__}")
    options = parser.parse_args()

    from .ui.theme import apply_theme
    from .ui.window import MainWindow, application_icon

    app = TensorApplication([sys.argv[0]])
    app.setApplicationName("TensorView")
    app.setApplicationDisplayName("TensorView")
    app.setApplicationVersion(__version__)
    apply_theme(app)
    app.setWindowIcon(QIcon(application_icon()))
    app.window = MainWindow(demo=not (options.empty or options.files or app.system_paths))
    app.window.show()
    pending = [*options.files, *(app.system_paths or [])]
    if pending:
        QTimer.singleShot(0, lambda: app.window.load_paths(pending))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
