"""Frozen application entry point and package smoke test."""

import multiprocessing
import os
import sys


def configure_bundle():
    sys.dont_write_bytecode = True

    from pathlib import Path

    from PySide6.QtCore import QStandardPaths

    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericCacheLocation)
    if not location:
        from tempfile import gettempdir

        location = gettempdir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(location) / "TensorView" / "numba"))


def check_package():
    import tempfile
    import time
    from pathlib import Path

    if sys.platform != "win32":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from tensorview.analysis import demo_files, project
    from tensorview.exporting import export_tensor
    from tensorview.loaders import open_file, require
    from tensorview.ui.theme import apply_theme
    from tensorview.ui.window import MainWindow

    app = QApplication([])
    apply_theme(app)
    window = MainWindow()
    window.show()
    try:
        deadline = time.monotonic() + 25
        while window.inspector.stats is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.015)
        assert window.inspector.stats is not None
        assert window.inspector.chart.png().width() == 2400
        source, _ = demo_files()
        tensor = source.tensors["encoder.projection.weight"]
        with tempfile.TemporaryDirectory() as temporary:
            for suffix in (".npy", ".npz", ".safetensors", ".h5", ".mat", ".pt"):
                output = Path(temporary) / f"weights{suffix}"
                export_tensor(tensor, output)
                result = open_file(output)
                assert next(iter(result.tensors.values())).shape == tensor.shape
                del result
        require("onnx", "scientific")
        embeddings = source.tensors["text.embeddings"]
        for method in ("PCA", "t-SNE", "UMAP"):
            result, indices = project(embeddings, method)
            assert result.shape == (len(indices), 2)
    finally:
        window.close()
        window.jobs.pool.waitForDone()
        app.processEvents()


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        configure_bundle()
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-package":
        import traceback
        from pathlib import Path

        file = Path(sys.argv[2])
        try:
            check_package()
        except Exception:
            file.write_text(traceback.format_exc(), encoding="utf-8")
            raise SystemExit(1) from None
        file.write_text("TensorView package checks passed.\n", encoding="utf-8")
    else:
        from tensorview.__main__ import main

        raise SystemExit(main())
