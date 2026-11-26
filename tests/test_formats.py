import json
from pathlib import Path

import numpy as np
import pytest

from tensorview.exporting import TENSOR_EXPORTS, export_report, export_tensor
from tensorview.loaders import open_file
from tensorview.model import TensorViewError, array_tensor


def test_numpy_readers_skip_objects(tmp_path):
    good = np.arange(48).reshape(2, 4, 6).transpose((2, 0, 1))
    path = tmp_path / "tensor.NPY"
    with path.open("wb") as stream:
        np.save(stream, good)
    np.testing.assert_array_equal(open_file(path).tensors["array"].read(), good)
    compressed = tmp_path / "tensor.npz"
    np.savez_compressed(compressed, sample=good, blocked=np.array([{"a": 1}], dtype=object))
    result = open_file(compressed)
    assert "blocked" not in result.tensors
    assert result.notes
    np.testing.assert_array_equal(result.tensors["sample"].preview(8), good.reshape(-1)[:8])
    np.testing.assert_array_equal(result.tensors["sample"].read(), good)


def test_safetensors_scalars_zero_and_extended_dtype(tmp_path):
    from safetensors.numpy import save_file

    path = tmp_path / "weights.safetensors"
    save_file({"scalar": np.array(3), "empty": np.empty((2, 0, 3)), "dense": np.arange(69)},
              str(path), metadata={"source": "test"})
    loaded = open_file(path)
    assert loaded.metadata["source"] == "test"
    assert loaded.tensors["scalar"].read().shape == ()
    assert loaded.tensors["empty"].read().shape == (2, 0, 3)
    assert loaded.tensors["dense"].preview(4).tolist() == [0, 1, 2, 3]
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file as torch_save_file

    path = tmp_path / "bfloat.safetensors"
    torch_save_file({"bfloat": torch.arange(12, dtype=torch.bfloat16)}, str(path))
    bfloat = open_file(path).tensors["bfloat"]
    assert bfloat.dtype_name == "bfloat16"
    assert bfloat.nbytes == 24
    np.testing.assert_array_equal(bfloat.read(), np.arange(12))


def test_torch_checkpoint_and_torchscript(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"weight": torch.arange(12).reshape(3, 4)}, "epoch": 12}, path)
    result = open_file(path)
    assert result.metadata["epoch"] == 12
    np.testing.assert_array_equal(result.tensors["state_dict.weight"].read((1, slice(None))), np.arange(4, 8))
    path = tmp_path / "scripted.pt"
    torch.jit.trace(torch.nn.Linear(2, 2), torch.zeros(1, 2)).save(str(path))
    with pytest.raises(TensorViewError, match="TorchScript"):
        open_file(path)


def test_hdf5_and_matlab_v73(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "model.hdf5"
    source = np.arange(24, dtype="float32").reshape(2, 3, 4)
    with h5py.File(path, "w") as target:
        target["encoder/weight"] = source
        target["encoder/skip"] = h5py.SoftLink("/absent")
        target.attrs["count"] = 42
    h5 = open_file(path)
    assert h5.metadata["count"] == 42
    np.testing.assert_array_equal(h5.tensors["encoder/weight"].read()[1], source[1])
    path = tmp_path / "data.mat"
    with h5py.File(path, "w") as target:
        target["x"] = source.T
        target["x"].attrs["MATLAB_class"] = np.bytes_("single")
        target["#refs#/hidden"] = source
    mat = open_file(path)
    assert list(mat.tensors) == ["x"]
    np.testing.assert_array_equal(mat.tensors["x"].read((1, slice(None), slice(None))), source[1])


def test_onnx_inline_and_external(tmp_path):
    onnx = pytest.importorskip("onnx")
    weight = np.arange(32, dtype=np.float32)
    model = onnx.helper.make_model(onnx.helper.make_graph(
        [], "sample", [], [], initializer=[onnx.numpy_helper.from_array(weight, name="weights")],
    ))
    inline = tmp_path / "inline.onnx"
    onnx.save(model, inline)
    np.testing.assert_array_equal(open_file(inline).tensors["weights"].read(), weight)
    external = tmp_path / "external.onnx"
    onnx.save_model(model, external, save_as_external_data=True, location="weights.data", size_threshold=0)
    np.testing.assert_array_equal(open_file(external).tensors["weights"].preview(7), weight[:7])
    model = onnx.load(external, load_external_data=False)
    for field in model.graph.initializer[0].external_data:
        if field.key == "location":
            field.value = "../weights.data"
    external.write_bytes(model.SerializeToString())
    result = open_file(external)
    assert not result.tensors and result.notes


def test_matlab_complex_and_logical(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    pytest.importorskip("h5py")
    path = tmp_path / "input.mat"
    data = np.array([[1 + 3j, 2 + 4j]], dtype=np.complex64)
    scipy_io.savemat(path, {"sample": data, "flag": np.array([[True, False]])})
    result = open_file(path)
    assert result.tensors["sample"].dtype == data.dtype
    assert result.tensors["flag"].read().dtype == np.bool_
    np.testing.assert_array_equal(result.tensors["sample"].read(), data)


@pytest.mark.parametrize("suffix,module", [
    (".npy", None), (".npz", None), (".safetensors", None),
    (".pt", "torch"), (".h5", "h5py"), (".mat", "scipy.io"),
])
def test_conversion_formats_round_trip(tmp_path, suffix, module):
    if module:
        pytest.importorskip(module)
    assert suffix in TENSOR_EXPORTS
    original = np.arange(24).reshape(2, 3, 4)
    source = array_tensor("layer.weight", original)
    output = tmp_path / f"converted{suffix}"
    export_tensor(source, output, dtype="float32", transpose=(2, 0, 1), shape=(4, -1))
    result = next(iter(open_file(output).tensors.values()))
    assert result.dtype == np.dtype("float32")
    np.testing.assert_array_equal(result.read(), original.transpose(2, 0, 1).reshape(4, 6))


def test_export_preserves_old_output_on_error_and_writes_csv_indices(tmp_path):
    source = array_tensor("input", np.arange(4).reshape(2, 2))
    output = tmp_path / "output.npy"
    output.write_bytes(b"old file")
    with pytest.raises(TensorViewError, match="must contain"):
        export_tensor(source, output, shape=(9,))
    assert output.read_bytes() == b"old file"
    with pytest.raises(TensorViewError, match="range"):
        export_tensor(array_tensor("input", np.array([0, 256])), output, dtype="uint8")
    assert output.read_bytes() == b"old file"
    csv = tmp_path / "out.csv"
    export_tensor(source, csv)
    assert csv.read_text().splitlines() == ["axis_0,axis_1,value", "0,0,0", "0,1,1", "1,0,2", "1,1,3"]


def test_report_formats_escape_user_controlled_names(tmp_path):
    for suffix in ("json", "csv", "html"):
        path = Path(tmp_path) / f"output.{suffix}"
        export_report(path, "Example", {"source": "<em>"}, [{"name": "=1+2", "result": "<script>"}])
        text = path.read_text()
        if suffix == "json":
            assert json.loads(text)["rows"][0]["name"] == "=1+2"
        elif suffix == "csv":
            assert "'=1+2" in text
        else:
            assert "<script>" not in text
            assert "&lt;script&gt;" in text
