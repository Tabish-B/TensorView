"""Local, non-executing readers for common scientific and ML file formats."""

import importlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from functools import lru_cache
from math import prod
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat
from safetensors import safe_open

from .model import (
    Tensor,
    TensorFile,
    TensorViewError,
    array_tensor,
    numeric_dtype,
    short_metadata,
    unique_name,
)

FORMATS = {
    ".npy": "NumPy array", ".npz": "NumPy archive", ".safetensors": "SafeTensors",
    ".pt": "PyTorch checkpoint", ".pth": "PyTorch checkpoint",
    ".h5": "HDF5", ".hdf5": "HDF5", ".onnx": "ONNX", ".mat": "MATLAB",
}
SAFE_DTYPES = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "U16": "uint16", "I16": "int16",
    "U32": "uint32", "I32": "int32", "U64": "uint64", "I64": "int64",
    "F16": "float16", "F32": "float32", "F64": "float64",
}
SAFE_ML_DTYPES = {
    "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
}
MEMORY_LIMIT = 512 * 1024**2


def require(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise TensorViewError(f"This feature requires {module}. Install tensorview[{extra}].") from error


def open_file(path: str | Path) -> TensorFile:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if not path.is_file():
        raise TensorViewError(f"File not found: {path}")
    if suffix not in READERS:
        raise TensorViewError(f"Unsupported file extension: {suffix or '(none)'}")
    result = TensorFile(path, path.name, FORMATS[suffix], {})
    try:
        READERS[suffix](path, result)
    except Exception as error:
        if isinstance(error, TensorViewError):
            raise
        raise TensorViewError(f"Could not read {path.name}: {str(error)[:900]}") from error
    return result


def _numpy(path: Path, result: TensorFile) -> None:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not numeric_dtype(array.dtype):
            raise TensorViewError(f"Only numerical arrays are supported. Found {array.dtype}.")
        result.tensors["array"] = array_tensor("array", array)
        return

    cache = tempfile.TemporaryDirectory(prefix="tensorview-npz-")
    with zipfile.ZipFile(path) as archive:
        for number, info in enumerate(archive.infolist()):
            if not info.filename.endswith(".npy") or info.is_dir():
                continue
            name = unique_name(result.tensors, info.filename[:-4])
            try:
                with archive.open(info) as stream:
                    version = npformat.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = npformat.read_array_header_1_0(stream)
                    elif version in {(2, 0), (3, 0)}:
                        # Versions 2 and 3 share the length encoding. Numeric descriptors are ASCII.
                        shape, _, dtype = npformat.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"unsupported NPY header version {version}")
                    expected_size = stream.tell() + prod(shape) * dtype.itemsize
                if not numeric_dtype(dtype):
                    raise ValueError(f"unsupported dtype {dtype}")
                if any(length < 0 for length in shape) or expected_size != info.file_size:
                    raise ValueError("shape does not match the member size")
            except (ValueError, EOFError, OSError, zipfile.BadZipFile) as error:
                result.notes.append(f"Skipped {name}: {error}")
                continue
            reader = _npz_reader(path, cache, number, info)
            result.tensors[name] = Tensor(name, shape, dtype, reader)


def _npz_reader(path, cache, number, info):
    destination = Path(cache.name) / f"{number}.npy"

    def read(index):
        if not destination.exists():
            if info.file_size > shutil.disk_usage(cache.name).free:
                raise TensorViewError(f"Insufficient temporary disk space for {info.filename}.")
            temporary = destination.with_suffix(".part")
            try:
                with zipfile.ZipFile(path) as archive, archive.open(info) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return np.load(destination, mmap_mode="r", allow_pickle=False)[index]

    return read


def _safetensors(path: Path, result: TensorFile) -> None:
    # SafeTensors validates the offsets and shape with its native parser before any map is created.
    with safe_open(path, framework="numpy") as source:
        result.metadata.update({key: short_metadata(value) for key, value in (source.metadata() or {}).items()})
        keys = list(source.keys())
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= 100_000_000:
            raise TensorViewError("Invalid SafeTensors header length.")
        header = json.loads(stream.read(length))

    @lru_cache(maxsize=1)
    def mapping():
        return np.memmap(path, dtype=np.uint8, mode="r")

    for name in keys:
        info = header[name]
        code = info["dtype"]
        if code in SAFE_DTYPES:
            storage_dtype = np.dtype(SAFE_DTYPES[code]).newbyteorder("<")
            dtype = storage_dtype
        elif code in SAFE_ML_DTYPES:
            try:
                import ml_dtypes

                storage_dtype = np.dtype(getattr(ml_dtypes, SAFE_ML_DTYPES[code])).newbyteorder("<")
                dtype = np.dtype("float32")
            except (ImportError, AttributeError):
                result.notes.append(f"Skipped {name}: {code} requires a compatible ml-dtypes installation.")
                continue
        else:
            result.notes.append(f"Skipped {name}: unsupported SafeTensors dtype {code}.")
            continue
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        if start < 0 or 8 + length + end > path.stat().st_size or end - start != prod(shape) * storage_dtype.itemsize:
            raise TensorViewError(f"Invalid SafeTensors bounds for {name}.")
        offset = 8 + length + start
        reader = _array_reader(mapping, shape, storage_dtype, offset, dtype)
        stored_name = SAFE_ML_DTYPES.get(code, "")
        result.tensors[name] = Tensor(name, shape, dtype, reader, stored_name, storage_dtype.itemsize)


def _array_reader(mapping, shape, storage_dtype, offset, dtype):
    def read(index):
        if 0 in shape:
            return np.empty(shape, dtype=dtype)[index]
        array = np.ndarray(shape, dtype=storage_dtype, buffer=mapping(), offset=offset)
        return np.asarray(array[index], dtype=dtype)

    return read


def _torch(path: Path, result: TensorFile) -> None:
    torch = require("torch", "torch")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if any(entry.filename.rsplit("/", 1)[-1] == "constants.pkl" for entry in archive.infolist()):
                raise TensorViewError("TorchScript archives are unsupported. Export a tensor state_dict.")
        mmap = True
    else:
        # Older tar serialization is intentionally rejected. Only use the restricted loader.
        with path.open("rb") as stream:
            if stream.read(1) != b"\x80":
                raise TensorViewError("Unsupported PyTorch serialization. Save a state_dict using torch.save.")
        mmap = False
    if not mmap and path.stat().st_size > MEMORY_LIMIT:
        raise TensorViewError("This checkpoint requires eager loading and exceeds 512 MiB. Use SafeTensors.")
    try:
        checkpoint = torch.load(path, weights_only=True, mmap=mmap, map_location="cpu")
    except Exception as error:
        raise TensorViewError(
            "The PyTorch weights-only loader could not read this checkpoint. "
            f"Save a plain state_dict or SafeTensors file. {str(error).splitlines()[0]}"
        ) from error

    active: set[int] = set()

    def visit(value, path_parts, depth=0):
        label = ".".join(path_parts) or "tensor"
        if depth > 64:
            return
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized or value.device.type != "cpu":
                result.notes.append(f"Skipped {label}: unsupported tensor layout or device.")
                return
            source = value.detach()
            try:
                dtype = source.new_empty(0).numpy().dtype
            except (TypeError, RuntimeError):
                if not source.is_floating_point() and not source.is_complex():
                    result.notes.append(f"Skipped {label}: unsupported PyTorch dtype {source.dtype}.")
                    return
                dtype = np.dtype("complex64" if source.is_complex() else "float32")
            name = unique_name(result.tensors, label)
            stored_dtype = str(source.dtype).removeprefix("torch.")
            result.tensors[name] = Tensor(
                name, tuple(source.shape), dtype, _torch_reader(source, dtype), stored_dtype, source.element_size(),
            )
        elif isinstance(value, (Mapping, list, tuple)) and id(value) not in active:
            active.add(id(value))
            children = value.items() if isinstance(value, Mapping) else enumerate(value)
            for key, child in children:
                visit(child, (*path_parts, str(key)), depth + 1)
            active.remove(id(value))
        elif isinstance(value, (str, bool, int, float)) and len(result.metadata) < 100:
            result.metadata[label] = short_metadata(value)

    visit(checkpoint, ())


def _torch_reader(source, dtype):
    def read(index):
        part = source[index].resolve_neg().resolve_conj()
        try:
            return part.numpy()
        except (TypeError, RuntimeError):
            return (part.cfloat() if dtype.kind == "c" else part.float()).numpy()

    return read


def _hdf5(path: Path, result: TensorFile) -> None:
    h5py = require("h5py", "scientific")
    matlab = path.suffix.lower() == ".mat"
    source = h5py.File(path, "r")
    if matlab:
        result.format = "MATLAB v7.3"

    def collect(name, item):
        if not isinstance(item, h5py.Dataset) or (matlab and name.split("/")[0].startswith("#")):
            return
        if item.is_virtual or item.external:
            result.notes.append(f"Skipped {name}: externally stored HDF5 dataset.")
            return
        if item.shape is None:
            result.notes.append(f"Skipped {name}: null HDF5 dataset.")
            return
        dtype = item.dtype
        complex_fields = matlab and dtype.names == ("real", "imag")
        if complex_fields:
            dtype = np.result_type(dtype["real"], dtype["imag"], np.complex64)
        if not numeric_dtype(dtype) or (matlab and item.attrs.get("MATLAB_class") == b"char"):
            result.notes.append(f"Skipped {name}: unsupported dataset dtype {item.dtype}.")
            return
        logical = matlab and item.attrs.get("MATLAB_class") == b"logical"
        if logical:
            dtype = np.dtype(bool)
        shape = tuple(reversed(item.shape)) if matlab else tuple(item.shape)
        result.tensors[name] = Tensor(name, shape, dtype, _hdf5_reader(item, matlab, complex_fields, logical))

    source.visititems(collect)
    for key in list(source.attrs)[:100]:
        attribute = source.attrs.get_id(key)
        if attribute.dtype.kind != "O" and attribute.get_storage_size() <= 2048:
            result.metadata[key] = short_metadata(np.asarray(source.attrs[key]).tolist())
    if not result.tensors:
        source.close()


def _hdf5_reader(dataset, transpose, complex_fields, logical):
    def read(index):
        index = (*index, *((slice(None),) * (dataset.ndim - len(index))))
        if transpose:
            index = tuple(reversed(index))
        value = np.asarray(dataset[index])
        if complex_fields:
            dtype = np.result_type(value.dtype["real"], value.dtype["imag"], np.complex64)
            complex_value = np.empty(value.shape, dtype=dtype)
            complex_value.real = value["real"]
            complex_value.imag = value["imag"]
            value = complex_value
        if logical:
            value = value.astype(bool, copy=False)
        return value.T if transpose else value

    return read


def _matlab(path: Path, result: TensorFile) -> None:
    h5py = require("h5py", "scientific")
    if h5py.is_hdf5(path):
        _hdf5(path, result)
        return
    scipy_io = require("scipy.io", "scientific")
    classes = {"single": "float32", "double": "float64", "logical": "bool"}
    classes.update({name: name for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")})
    @lru_cache(maxsize=1)
    def load(name):
        return scipy_io.loadmat(path, variable_names=[name])[name]

    for name, shape, kind in scipy_io.whosmat(path):
        if kind not in classes:
            result.notes.append(f"Skipped {name}: unsupported MATLAB class {kind}.")
            continue
        dtype = np.dtype(classes[kind])
        if prod(shape) * dtype.itemsize * (2 if dtype.kind == "f" else 1) > MEMORY_LIMIT:
            result.notes.append(f"Skipped {name}: MATLAB variable exceeds the eager-read limit.")
            continue
        # whosmat reports the MATLAB class, independent of whether float data is complex.
        array = load(name)
        dtype = array.dtype if array.dtype.kind == "c" else dtype
        def read(index, name=name, dtype=dtype):
            return np.asarray(load(name)[index], dtype=dtype)

        result.tensors[name] = Tensor(name, array.shape, dtype, read)
    load.cache_clear()


def _onnx(path: Path, result: TensorFile) -> None:
    onnx = require("onnx", "scientific")
    if path.stat().st_size > MEMORY_LIMIT:
        raise TensorViewError("The ONNX protobuf exceeds 512 MiB. Store weights in an external data file.")
    model = onnx.load(path, load_external_data=False)
    result.metadata.update({
        "Producer": model.producer_name,
        "Producer version": model.producer_version,
        "IR version": model.ir_version,
        "Graph": model.graph.name,
        "Nodes": len(model.graph.node),
        "Opsets": ", ".join(f"{op.domain or 'ai.onnx'} {op.version}" for op in model.opset_import),
    })
    result.metadata.update({short_metadata(x.key): short_metadata(x.value) for x in model.metadata_props})

    @lru_cache(maxsize=8)
    def external_mapping(location):
        return np.memmap(location, dtype=np.uint8, mode="r")

    def add(proto, label):
        name = unique_name(result.tensors, label)
        try:
            stored_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(proto.data_type))
            extended_float = stored_dtype.kind == "V" and (
                "float" in stored_dtype.name
            )
            if not numeric_dtype(stored_dtype) and not extended_float:
                raise TensorViewError(f"unsupported ONNX dtype {stored_dtype.name}")
            dtype = np.dtype("float32") if extended_float else stored_dtype
            if proto.data_location == onnx.TensorProto.EXTERNAL:
                reader = _onnx_external(path, proto, stored_dtype, dtype, external_mapping)
            else:
                reader = _onnx_reader(onnx, proto, dtype)
            result.tensors[name] = Tensor(
                name, tuple(proto.dims), dtype, reader, stored_dtype.name, stored_dtype.itemsize,
            )
        except (TypeError, ValueError, OSError, KeyError) as error:
            result.notes.append(f"Skipped {name}: {error}")

    def graph_tensors(graph, prefix=""):
        for index, proto in enumerate(graph.initializer):
            add(proto, prefix + (proto.name or f"initializer_{index}"))
        for number, node in enumerate(graph.node):
            label = f"{prefix}{node.name or node.op_type + '_' + str(number)}/"
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    add(attr.t, label + attr.name)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for index, proto in enumerate(attr.tensors):
                        add(proto, f"{label}{attr.name}/{index}")
                elif attr.type == onnx.AttributeProto.GRAPH:
                    graph_tensors(attr.g, f"{label}{attr.name}/")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for index, subgraph in enumerate(attr.graphs):
                        graph_tensors(subgraph, f"{label}{attr.name}/{index}/")
        if graph.sparse_initializer:
            result.notes.append(f"Skipped {len(graph.sparse_initializer)} sparse ONNX initializers.")

    graph_tensors(model.graph)


def _onnx_reader(onnx, proto, dtype):
    @lru_cache(maxsize=1)
    def load():
        return onnx.numpy_helper.to_array(proto).astype(dtype, copy=False)

    return lambda index: load()[index]


def _onnx_external(path, proto, stored_dtype, dtype, external_mapping):
    # External data is read only from local, ordinary files directly alongside this model.
    fields = {entry.key: entry.value for entry in proto.external_data}
    location = fields.get("location", "")
    if not location or location in {".", ".."} or any(char in location for char in "/\\:\x00"):
        raise TensorViewError("external weights must be in a file alongside the ONNX model")
    external = path.parent / location
    if external.is_symlink() or not external.is_file() or external.stat().st_nlink != 1:
        raise TensorViewError("external weights must be a regular file without links")
    if "float4" in stored_dtype.name:
        raise TensorViewError("packed external ONNX weights are unsupported")
    offset = int(fields.get("offset", 0))
    shape = tuple(proto.dims)
    required = prod(shape) * stored_dtype.itemsize
    length = int(fields.get("length", 0)) or required
    if min(shape, default=0) < 0 or offset < 0 or length != required or offset + length > external.stat().st_size:
        raise TensorViewError("invalid external weight bounds")
    return _array_reader(
        lambda: external_mapping(external), shape, stored_dtype.newbyteorder("<"), offset, dtype,
    )


READERS = {
    ".npy": _numpy, ".npz": _numpy, ".safetensors": _safetensors, ".pt": _torch, ".pth": _torch,
    ".h5": _hdf5, ".hdf5": _hdf5, ".mat": _matlab, ".onnx": _onnx,
}
