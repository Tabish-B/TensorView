"""Atomic tensor conversions and portable analysis reports."""

import csv
import html
import json
import math
import mmap
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from math import prod
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .analysis import Difference, Finding, Statistics
from .loaders import MEMORY_LIMIT, require
from .model import Tensor, TensorFile, TensorViewError, block_indices, numeric_dtype

TENSOR_EXPORTS = {
    ".npy": "NumPy array", ".npz": "Compressed NumPy archive", ".safetensors": "SafeTensors",
    ".pt": "PyTorch state dictionary", ".h5": "HDF5 dataset", ".mat": "MATLAB array", ".csv": "CSV values",
}


@contextmanager
def atomic_destination(path: Path) -> Iterator[Path]:
    handle, name = tempfile.mkstemp(prefix=".tensorview-", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_tensor(
    tensor: Tensor, path: str | Path, *, dtype: str = "", shape: tuple[int, ...] | None = None,
    transpose: tuple[int, ...] | None = None,
):
    path = Path(path).expanduser().absolute()
    suffix = path.suffix.lower()
    if suffix not in TENSOR_EXPORTS:
        raise TensorViewError(f"Unsupported output format: {suffix}")
    target_dtype = np.dtype(dtype) if dtype else tensor.dtype.newbyteorder("=")
    if not numeric_dtype(target_dtype):
        raise TensorViewError("Choose a numeric output dtype.")
    axes = tuple(range(len(tensor.shape))) if transpose is None else transpose
    if sorted(axes) != list(range(len(tensor.shape))):
        raise TensorViewError("Transpose axes must be a permutation of the input axes.")
    source_shape = tuple(tensor.shape[axis] for axis in axes)
    shape = _resolve_shape(shape, tensor.size) if shape is not None else source_shape
    if suffix in {".pt", ".safetensors", ".mat"} and tensor.size * target_dtype.itemsize > MEMORY_LIMIT:
        raise TensorViewError("Use NPY, NPZ, HDF5 or CSV for exports larger than 512 MiB.")

    with atomic_destination(path) as destination:
        if suffix == ".npy":
            _stage(tensor, destination, target_dtype, axes, source_shape, shape)
        else:
            with tempfile.TemporaryDirectory(prefix="tensorview-export-", dir=path.parent) as workspace:
                staged = Path(workspace) / "tensor.npy"
                offset = _stage(tensor, staged, target_dtype, axes, source_shape, shape)
                with staged.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
                    array = np.ndarray(shape, dtype=target_dtype, buffer=mapped, offset=offset)
                    _write_array(destination, suffix, array, tensor.name)


def _resolve_shape(shape, size):
    if shape.count(-1) > 1 or any(length < -1 for length in shape):
        raise TensorViewError("Reshape dimensions must be nonnegative, with at most one inferred axis (-1).")
    if -1 in shape:
        known = prod(length for length in shape if length != -1)
        if not known or size % known:
            raise TensorViewError("The inferred dimension does not match the element count.")
        shape = tuple(size // known if length == -1 else length for length in shape)
    if prod(shape) != size:
        raise TensorViewError(f"The new shape must contain {size:,} elements.")
    return shape


def _stage(tensor, path, dtype, axes, source_shape, destination_shape):
    inverse = np.argsort(axes)
    header = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": destination_shape,
    }
    with path.open("wb") as stream:
        np.lib.format.write_array_header_2_0(stream, header)
        offset = stream.tell()
        for index in block_indices(source_shape):
            logical_index = tuple(index[axis] for axis in inverse)
            remaining = [axis for axis, item in enumerate(logical_index) if isinstance(item, slice)]
            permutation = tuple(remaining.index(axes[i]) for i, item in enumerate(index) if isinstance(item, slice))
            values = tensor.read(logical_index).transpose(permutation).reshape(-1)
            if values.dtype.kind == "c" and dtype.kind != "c":
                raise TensorViewError("Choose a complex dtype to preserve the imaginary component.")
            if dtype.kind in "iu":
                _check_integer_range(values, dtype)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    converted = values.astype(dtype, copy=False)
            except FloatingPointError as error:
                raise TensorViewError(f"Values exceed the range of {dtype}.") from error
            stream.write(converted.tobytes())
    return offset


def _check_integer_range(values, dtype):
    info = np.iinfo(dtype)
    if values.dtype.kind == "f":
        truncated = np.trunc(values.astype(np.float64, copy=False))
        invalid = ~np.isfinite(truncated) | (truncated < float(info.min)) | (truncated >= float(int(info.max) + 1))
    elif values.dtype.kind in "iu" and not np.can_cast(values.dtype, dtype, casting="safe"):
        source = np.iinfo(values.dtype)
        lower = max(int(info.min), int(source.min))
        upper = min(int(info.max), int(source.max))
        invalid = (values < lower) | (values > upper)
    else:
        return
    if np.any(invalid):
        raise TensorViewError(f"Values exceed the range of {dtype} or include NaN or infinity.")


def _write_array(path, suffix, array, name):
    if suffix == ".npz":
        # Use a stream so a tensor called "file" cannot collide with NumPy's arguments.
        import zipfile

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            label = name if name not in {".", ".."} else "tensor"
            with archive.open(label + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
    elif suffix == ".safetensors":
        if array.dtype.kind == "c":
            raise TensorViewError("Use NPY, NPZ, HDF5, MATLAB, PyTorch or CSV for complex values.")
        save_file({name: array}, str(path))
    elif suffix == ".pt":
        torch = require("torch", "torch")
        torch.save({name: torch.from_numpy(array)}, path)
    elif suffix == ".h5":
        h5py = require("h5py", "scientific")
        with h5py.File(path, "w") as file:
            label = "/".join(part if part not in {"", ".", ".."} else "_" for part in name.split("/"))
            dataset = file.create_dataset(label, shape=array.shape, dtype=array.dtype)
            for index in block_indices(array.shape):
                dataset[index] = array[index]
    elif suffix == ".mat":
        # MATLAB identifies variables with a leading letter and alphanumeric/underscore characters.
        import re

        scipy_io = require("scipy.io", "scientific")
        label = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if not label or not label[0].isalpha():
            label = "tensor_" + label
        with path.open("wb") as stream:
            scipy_io.savemat(stream, {label[:63]: array})
    elif suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([*(f"axis_{i}" for i in range(array.ndim)), "value"])
            for index, value in zip(np.ndindex(array.shape), array.flat, strict=True):
                writer.writerow([*index, value.item()])


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def tensor_report(source: TensorFile, tensor: Tensor, stats: Statistics) -> tuple[str, dict, list[dict]]:
    metrics = {key: getattr(stats, key) for key in stats.__dataclass_fields__ if key != "sample"}
    metrics["sparsity"] = stats.sparsity
    percentiles = metrics.pop("percentiles")
    metrics.update({f"p{key}": value for key, value in percentiles.items()})
    return "Tensor inspection", {
        "file": source.label, "tensor": tensor.name, "shape": tensor.shape, "dtype": tensor.dtype_name,
        "stored_bytes": tensor.nbytes, "format": source.format,
    }, [{"metric": key, "value": value} for key, value in metrics.items()]


def validation_report(source: TensorFile, findings: list[Finding]) -> tuple[str, dict, list[dict]]:
    return "Numerical validation", {
        "file": source.label, "tensors": len(source.tensors), "findings": len(findings),
    }, [asdict(item) for item in findings]


def comparison_report(
    left: TensorFile, right: TensorFile, differences: list[Difference],
) -> tuple[str, dict, list[dict]]:
    return "Checkpoint comparison", {
        "baseline": left.label, "candidate": right.label,
        "baseline_elements": left.size, "candidate_elements": right.size,
    }, [asdict(item) for item in differences]


def export_report(path: str | Path, title: str, info: dict, rows: list[dict]):
    path = Path(path).expanduser().absolute()
    suffix = path.suffix.lower()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    info, rows = _json_safe(info), _json_safe(rows)
    with atomic_destination(path) as destination, destination.open("w", encoding="utf-8", newline="") as output:
        if suffix == ".json":
            json.dump({"report": title, "summary": info, "rows": rows}, output, indent=2, allow_nan=False)
            output.write("\n")
        elif suffix == ".csv":
            if fields:
                context = {f"summary_{key}": _csv_cell(value) for key, value in info.items()}
                writer = csv.DictWriter(output, fieldnames=[*context, *fields])
                writer.writeheader()
                for row in rows:
                    writer.writerow({**context, **{key: _csv_cell(value) for key, value in row.items()}})
            else:
                csv.writer(output).writerows([["field", "value"], *((key, _csv_cell(value)) for key, value in info.items())])
        elif suffix == ".html":
            _html_report(output, title, info, rows, fields)
        else:
            raise TensorViewError("Report format must be JSON, CSV, or HTML.")


def _csv_cell(value):
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def _html_report(output, title, info, rows, fields):
    def escape(value):
        return html.escape("" if value is None else str(value))
    output.write(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1"><title>{escape(title)} · TensorView</title>'
        '<style>body{font:15px system-ui;color:#142137;margin:40px auto;padding:0 28px;max-width:1400px}'
        'table{border-collapse:collapse;text-align:left;width:100%;font-size:13px}dt{font-weight:650}'
        'dd{margin:4px 0 12px}td,th{padding:8px 12px;border-bottom:1px solid #d3ddea}'
        'th{background:#edf3fa}section{overflow:auto}</style>'
        f'<main><h1>{escape(title)}</h1><dl>'
    )
    for key, value in info.items():
        output.write(f"<dt>{escape(key.replace('_', ' ').title())}</dt><dd>{escape(value)}</dd>")
    output.write("</dl>")
    if fields:
        output.write("<section><table><thead><tr>")
        output.write("".join(f"<th>{escape(key.replace('_', ' ').title())}</th>" for key in fields))
        output.write("</tr></thead><tbody>")
        for row in rows:
            output.write("<tr>" + "".join(f"<td>{escape(row.get(key))}</td>" for key in fields) + "</tr>")
        output.write("</tbody></table></section>")
    output.write("</main></html>\n")
