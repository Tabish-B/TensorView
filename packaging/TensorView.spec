from pathlib import Path
from sys import platform

from PyInstaller.utils.hooks import copy_metadata

root = Path(SPECPATH).parent
assets = root / "packaging"
icon = assets / ("TensorView.icns" if platform == "darwin" else "TensorView.ico")
analysis = Analysis(
    [str(assets / "launch.py")],
    pathex=[str(root)],
    binaries=[],
    datas=copy_metadata("umap-learn"),
    hiddenimports=["h5py", "scipy.io", "onnx", "torch", "sklearn.decomposition", "sklearn.manifold", "umap"],
    excludes=["PyQt5", "PyQt6", "PySide2", "matplotlib", "IPython", "pytest", "torchaudio", "torchvision",
              "tensorflow", "pandas", "pyarrow", "jax", "transformers", "datasets", "accelerate"],
    # Numba caches functions by source location. Keep the JIT-decorated modules available on disk.
    module_collection_mode={"umap": "py", "pynndescent": "py"},
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True, name="TensorView",
    debug=False, strip=False, upx=False, console=False, argv_emulation=False,
    icon=str(icon) if platform == "win32" else None,
)
bundle = COLLECT(
    executable, analysis.binaries, analysis.datas, strip=False, upx=False, name="TensorView",
)
if platform == "darwin":
    app = BUNDLE(
        bundle, name="TensorView.app", icon=str(icon), bundle_identifier="org.tensorview.desktop",
        version="1.0.0",
        info_plist={
            "CFBundleName": "TensorView",
            "CFBundleDisplayName": "TensorView",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Tensor or checkpoint",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "CFBundleTypeExtensions": ["npy", "npz", "safetensors", "pt", "pth", "h5", "hdf5", "onnx", "mat"],
            }],
        },
    )
