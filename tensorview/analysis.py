"""Numerical scans, visualization data, and checkpoint differences."""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import frexp, inf, ldexp, sqrt

import numpy as np

from .loaders import require
from .model import Tensor, TensorFile, TensorViewError

Progress = Callable[[int, int], None]
SAMPLE_LIMIT = 65_536
PLOT_LIMIT = 384


@dataclass(slots=True)
class Statistics:
    count: int
    finite: int
    zeros: int
    nan: int
    positive_inf: int
    negative_inf: int
    minimum: int | float | None
    maximum: int | float | None
    mean: float | None
    median: float | None
    std: float | None
    percentiles: dict[str, float]
    sampled: bool
    complex: bool
    sample: np.ndarray = field(repr=False)

    @property
    def sparsity(self):
        return self.zeros / self.count if self.count else 0.0

    @property
    def infinite(self):
        return self.positive_inf + self.negative_inf


class _Moments:
    def __init__(self):
        self.count = 0
        self.scale = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, values):
        if not values.size:
            return
        maximum = float(np.max(np.abs(values)))
        scale = max(self.scale, maximum)
        if scale:
            previous_ratio = self.scale / scale
            self.mean *= previous_ratio
            self.m2 *= previous_ratio**2
            normalized = values / scale
            mean = float(normalized.mean())
            m2 = float(np.square(normalized - mean).sum())
        else:
            mean = m2 = 0.0
        total = self.count + values.size
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * (self.count * values.size / total)
        self.mean += delta * (values.size / total)
        self.scale = scale
        self.count = total


def _real_values(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = np.abs(array.astype(np.complex128))
        return np.minimum(magnitude, np.finfo(np.float64).max, where=np.isfinite(array), out=magnitude)
    return array.astype(np.float64, copy=False)


def statistics(tensor: Tensor, progress: Progress | None = None) -> Statistics:
    moments = _Moments()
    nan = positive_inf = negative_inf = zeros = 0
    minimum, maximum = np.inf, -np.inf
    sample_indices = np.linspace(0, tensor.size - 1, min(tensor.size, SAMPLE_LIMIT), dtype=np.int64)
    samples = []
    offset = 0
    sample_start = 0
    for block in tensor.blocks():
        if progress:
            progress(offset, tensor.size)
        samples_in_block = int(np.searchsorted(sample_indices, offset + block.size))
        is_nan = np.isnan(block)
        is_finite = np.isfinite(block)
        values = _real_values(block)
        nan += int(np.count_nonzero(is_nan))
        zeros += int(np.count_nonzero(block == 0))
        if tensor.dtype.kind == "c":
            positive_inf += int(np.count_nonzero(~is_finite & ~is_nan))
        else:
            positive_inf += int(np.count_nonzero(np.isposinf(values)))
            negative_inf += int(np.count_nonzero(np.isneginf(values)))
        finite = values[is_finite]
        if finite.size:
            extrema = block[is_finite] if tensor.dtype.kind in "bui" else finite
            minimum = min(minimum, extrema.min().item())
            maximum = max(maximum, extrema.max().item())
            moments.add(finite)
        chosen = sample_indices[sample_start:samples_in_block] - offset
        if chosen.size:
            samples.append(values[chosen[is_finite[chosen]]])
        sample_start = samples_in_block
        offset += block.size
    sample = np.concatenate(samples) if samples else np.empty(0)
    quantiles = (1, 5, 25, 50, 75, 95, 99)
    if sample.size:
        scale = float(np.max(np.abs(sample))) or 1.0
        percentiles = {
            str(q): float(value * scale)
            for q, value in zip(quantiles, np.percentile(sample / scale, quantiles), strict=True)
        }
    else:
        percentiles = {}
    return Statistics(
        tensor.size, moments.count, zeros, nan, positive_inf, negative_inf,
        minimum if moments.count else None, maximum if moments.count else None,
        moments.mean * moments.scale if moments.count else None, percentiles.get("50"),
        min(1.0, (moments.m2 / moments.count) ** 0.5) * moments.scale if moments.count else None,
        percentiles, len(sample_indices) < tensor.size, tensor.dtype.kind == "c", sample,
    )


@dataclass(slots=True)
class Finding:
    level: str
    tensor: str
    check: str
    detail: str


def _precision(tensor: Tensor) -> str:
    return tensor.dtype_name


def validate(source: TensorFile, progress: Progress | None = None) -> list[Finding]:
    findings = []
    floating_types = {_precision(t) for t in source.tensors.values() if t.dtype.kind in "fc"}
    if len(floating_types) > 1:
        findings.append(Finding("Info", "File", "Mixed floating-point precision", ", ".join(sorted(floating_types))))
    if not source.tensors:
        findings.append(Finding("Info", "File", "No supported tensors", "No numerical arrays were discovered."))
    for note in source.notes:
        findings.append(Finding("Info", "File", "Reader note", note))
    total = len(source.tensors)
    for index, tensor in enumerate(source.tensors.values(), 1):
        if progress:
            progress(index - 1, total)
        def update(_completed, _total, index=index):
            if progress:
                progress(index - 1, total)

        try:
            stats = statistics(tensor, update if progress else None)
        except Exception as error:
            findings.append(Finding("Error", tensor.name, "Read failure", str(error)))
            continue
        if stats.nan:
            findings.append(Finding("Error", tensor.name, "NaN values", f"{stats.nan:,} values"))
        if stats.infinite:
            findings.append(Finding("Error", tensor.name, "Infinite values", f"{stats.infinite:,} values"))
        if tensor.size == 0:
            findings.append(Finding("Info", tensor.name, "Empty tensor", "The tensor has no elements."))
        elif stats.zeros == tensor.size:
            findings.append(Finding("Review", tensor.name, "All zero", f"All {tensor.size:,} values are zero."))
        elif tensor.size > 1 and stats.finite == tensor.size and stats.minimum == stats.maximum and not stats.complex:
            findings.append(Finding("Review", tensor.name, "Constant tensor", "All values have the same numerical value."))
        elif stats.sparsity >= 0.99:
            findings.append(Finding("Review", tensor.name, "High sparsity", f"{stats.sparsity:.2%} of values are zero."))
        if stats.minimum is not None and max(abs(stats.minimum), abs(stats.maximum)) > 1e6:
            findings.append(Finding("Review", tensor.name, "Large values", "At least one finite magnitude exceeds 1,000,000."))
    if progress:
        progress(total, total)
    return findings


@dataclass(slots=True)
class Difference:
    name: str
    status: str
    shape_a: tuple[int, ...] | None
    shape_b: tuple[int, ...] | None
    dtype_a: str | None
    dtype_b: str | None
    count_a: int = 0
    count_b: int = 0
    changed_values: int | None = None
    finite_pairs: int | None = None
    mean_absolute: float | None = None
    max_absolute: float | None = None
    relative_l2: float | None = None
    cosine: float | None = None
    detail: str = ""


def _equal(left, right):
    if left.dtype.kind == "c" or right.dtype.kind == "c":
        return _equal(left.real, right.real) & _equal(left.imag, right.imag)
    left_integer = left.dtype.kind in "bui"
    right_integer = right.dtype.kind in "bui"
    if left_integer and right_integer:
        if left.dtype.kind == "u" and right.dtype.kind == "i":
            return (right >= 0) & np.equal(left, right.astype(np.uint64))
        if left.dtype.kind == "i" and right.dtype.kind == "u":
            return (left >= 0) & np.equal(left.astype(np.uint64), right)
    if left_integer != right_integer:
        integer, floating = (left, right) if left_integer else (right, left)
        floating = floating.astype(np.float64, copy=False)
        lower, upper = (0, 2) if integer.dtype.kind == "b" else (
            int(np.iinfo(integer.dtype).min), int(np.iinfo(integer.dtype).max) + 1,
        )
        valid = np.isfinite(floating) & (floating >= lower) & (floating < upper)
        safe = np.where(valid, floating, 0)
        return valid & (safe == np.trunc(safe)) & np.equal(integer, safe.astype(integer.dtype))
    return np.equal(left, right) | (np.isnan(left) & np.isnan(right))


def _component_scale(values):
    scale = float(np.max(np.abs(values.real)))
    if np.iscomplexobj(values):
        scale = max(scale, float(np.max(np.abs(values.imag))))
    return scale


def _scaled(values, scale):
    divisor = scale or 1.0
    if np.iscomplexobj(values):
        # Scale components separately to avoid overflow in complex division by subnormal scalars.
        return values.real / divisor + 1j * (values.imag / divisor)
    return values / divisor


class _Distance:
    def __init__(self):
        self.left_scale = self.right_scale = self.delta_scale = 0.0
        self.left_squared = self.right_squared = self.delta_squared = 0.0
        self.absolute = self.maximum = self.dot = 0.0

    def add(self, left, right, raw_delta):
        left_scale = max(self.left_scale, _component_scale(left))
        right_scale = max(self.right_scale, _component_scale(right))
        left_ratio = self.left_scale / left_scale if left_scale else 1.0
        right_ratio = self.right_scale / right_scale if right_scale else 1.0

        if raw_delta is None:
            with np.errstate(over="ignore"):
                raw_delta = left - right
        if np.isfinite(raw_delta).all():
            delta_scale = max(self.delta_scale, _component_scale(raw_delta))
            delta = np.abs(_scaled(raw_delta, delta_scale))
        else:
            # A finite pair can have a difference larger than float64 can hold.
            delta_scale = max(self.delta_scale, left_scale, right_scale)
            delta = np.abs(_scaled(left, delta_scale) - _scaled(right, delta_scale))
        delta_ratio = self.delta_scale / delta_scale if delta_scale else 1.0

        left, right = _scaled(left, left_scale), _scaled(right, right_scale)
        self.left_squared = self.left_squared * left_ratio**2 + float(np.square(np.abs(left)).sum())
        self.right_squared = self.right_squared * right_ratio**2 + float(np.square(np.abs(right)).sum())
        self.dot = self.dot * (left_ratio * right_ratio) + float(np.vdot(left, right).real)
        self.delta_squared = self.delta_squared * delta_ratio**2 + float(np.square(delta).sum())
        self.absolute = self.absolute * delta_ratio + float(delta.sum())
        self.maximum = max(self.maximum * delta_ratio, float(delta.max()))
        self.left_scale, self.right_scale, self.delta_scale = left_scale, right_scale, delta_scale

    def relative_l2(self):
        if not self.delta_scale:
            return 0.0
        left_fraction, left_exponent = frexp(self.left_scale)
        delta_fraction, delta_exponent = frexp(self.delta_scale)
        fraction = (delta_fraction / left_fraction) * sqrt(self.delta_squared / self.left_squared)
        try:
            return ldexp(fraction, delta_exponent - left_exponent)
        except OverflowError:
            return inf


def _integer_delta(a, b):
    # Offset signed values into monotonic unsigned space so nearby 64-bit integers stay distinct.
    if a.dtype.kind == b.dtype.kind == "i":
        offset = np.uint64(1 << 63)
        a = a.astype(np.int64).view(np.uint64) ^ offset
        b = b.astype(np.int64).view(np.uint64) ^ offset
    elif "i" in (a.dtype.kind, b.dtype.kind):
        signed, unsigned = (a, b) if a.dtype.kind == "i" else (b, a)
        negative = signed < 0
        a = np.where(negative, 0, signed).astype(np.uint64)
        b = unsigned.astype(np.uint64)
        delta = (np.maximum(a, b) - np.minimum(a, b)).astype(np.float64)
        delta[negative] += np.abs(signed[negative].astype(np.float64))
        return delta
    else:
        a, b = a.astype(np.uint64), b.astype(np.uint64)
    return (np.maximum(a, b) - np.minimum(a, b)).astype(np.float64)


def _integer_float_delta(integer, floating):
    real = floating.real.astype(np.float64, copy=False)
    delta = integer.astype(np.float64) - real
    if integer.dtype.itemsize == 8:
        wide = (integer > 2**53) | (integer < -(2**53)) if integer.dtype.kind == "i" else integer > 2**53
        if np.any(wide):
            # Both halves convert exactly. Subtract the float before adding the low bits.
            values = integer[wide]
            high = (values >> 32).astype(np.float64) * 2**32
            low = (values & 0xffffffff).astype(np.float64)
            delta[wide] = (high - real[wide]) + low
    if np.iscomplexobj(floating):
        complex_delta = delta.astype(np.complex128)
        complex_delta.imag = -floating.imag
        return complex_delta
    return delta


def _precise_delta(a, b):
    a_integer = a.dtype.kind in "bui"
    b_integer = b.dtype.kind in "bui"
    if a_integer and b_integer:
        return _integer_delta(a, b)
    if a_integer or b_integer:
        return _integer_float_delta(a, b) if a_integer else _integer_float_delta(b, a)
    return None

def compare_tensor(
    name: str, left: Tensor | None, right: Tensor | None, poll: Callable[[], None] | None = None,
) -> Difference:
    result = Difference(
        name, "", left.shape if left else None, right.shape if right else None,
        left.dtype_name if left else None, right.dtype_name if right else None,
        left.size if left else 0, right.size if right else 0,
    )
    if left is None or right is None:
        result.status = "Added" if left is None else "Removed"
        return result
    if left.shape != right.shape:
        result.status = "Shape changed"
        return result
    distance = _Distance()
    changed = pairs = 0
    dtype = np.complex128 if "c" in (left.dtype.kind, right.dtype.kind) else np.float64
    for a, b in zip(left.blocks(), right.blocks(), strict=True):
        if poll:
            poll()
        changed += int(np.count_nonzero(~_equal(a, b)))
        finite = np.isfinite(a) & np.isfinite(b)
        a, b = a[finite], b[finite]
        if a.size:
            pairs += a.size
            distance.add(a.astype(dtype, copy=False), b.astype(dtype, copy=False), _precise_delta(a, b))
    result.changed_values = changed
    result.finite_pairs = pairs
    result.status = "Modified" if changed else "Identical"
    if result.dtype_a != result.dtype_b:
        result.status = "Modified" if changed else "Dtype changed"
        result.detail = "Data type changed."
    if pairs:
        result.mean_absolute = distance.absolute / pairs * distance.delta_scale
        result.max_absolute = distance.maximum * distance.delta_scale
        if distance.left_squared:
            result.relative_l2 = distance.relative_l2()
        if not changed and not distance.left_squared:
            result.relative_l2 = 0.0
        if distance.left_squared and distance.right_squared:
            divisor = distance.left_squared**0.5 * distance.right_squared**0.5
            result.cosine = float(np.clip(distance.dot / divisor, -1.0, 1.0))
    return result


def compare(left: TensorFile, right: TensorFile, progress: Progress | None = None) -> list[Difference]:
    names = sorted(left.tensors.keys() | right.tensors.keys(), key=str.casefold)
    results = []
    for index, name in enumerate(names):
        if progress:
            progress(index, len(names))
        def poll(index=index):
            if progress:
                progress(index, len(names))

        try:
            results.append(compare_tensor(name, left.tensors.get(name), right.tensors.get(name), poll if progress else None))
        except Exception as error:
            a, b = left.tensors.get(name), right.tensors.get(name)
            results.append(Difference(
                name, "Read error", a.shape if a else None, b.shape if b else None,
                a.dtype_name if a else None, b.dtype_name if b else None,
                a.size if a else 0, b.size if b else 0, detail=str(error),
            ))
    if progress:
        progress(len(names), len(names))
    return results


def plot_slice(tensor: Tensor, axes: tuple[int, ...], fixed: dict[int, int]):
    if tensor.size == 0:
        return np.empty(0), (), ()
    index = []
    steps = []
    lengths = []
    for axis, length in enumerate(tensor.shape):
        if axis in axes:
            limit = 4096 if len(axes) == 1 else PLOT_LIMIT
            step = max(1, (length + limit - 1) // limit)
            steps.append(step)
            lengths.append(length)
            index.append(slice(0, length, step))
        else:
            index.append(fixed.get(axis, 0))
    values = _real_values(tensor.read(tuple(index)))
    if len(axes) == 2 and axes[0] > axes[1]:
        return values.T, tuple(reversed(steps)), tuple(reversed(lengths))
    return values, tuple(steps), tuple(lengths)


def project(tensor: Tensor, method: str) -> tuple[np.ndarray, np.ndarray]:
    if len(tensor.shape) != 2 or min(tensor.shape) < 2:
        raise TensorViewError("Select a 2D tensor with at least two rows and two columns.")
    rows, columns = tensor.shape
    row_step = max(1, (rows + 1999) // 2000)
    col_step = max(1, (columns + 511) // 512)
    values = _real_values(tensor.read((slice(None, None, row_step), slice(None, None, col_step))))
    indices = np.arange(0, rows, row_step)
    valid = np.all(np.isfinite(values), axis=1)
    indices, values = indices[valid], values[valid]
    minimum = {"PCA": 2, "t-SNE": 3, "UMAP": 4}.get(method, 2)
    if len(values) < minimum:
        raise TensorViewError("Too few sampled rows with only finite values for this projection.")
    scale = np.max(np.abs(values))
    if scale == 0 or np.all(values == values[0]):
        raise TensorViewError("Embedding projection requires variation across rows.")
    values = (values / scale).astype(np.float32)
    if method == "PCA":
        decomposition = require("sklearn.decomposition", "embeddings")
        result = decomposition.PCA(n_components=2, random_state=42).fit_transform(values)
    elif method == "t-SNE":
        manifold = require("sklearn.manifold", "embeddings")
        result = manifold.TSNE(
            n_components=2, perplexity=min(30.0, (len(values) - 1) / 3), random_state=42, n_jobs=1,
        ).fit_transform(values)
    elif method == "UMAP":
        umap = require("umap", "embeddings")
        result = umap.UMAP(n_components=2, n_neighbors=min(15, len(values) - 1), random_state=42, n_jobs=1).fit_transform(values)
    else:
        raise TensorViewError(f"Unknown projection: {method}")
    return result, indices


def demo_files() -> tuple[TensorFile, TensorFile]:
    from .model import array_tensor

    rng = np.random.default_rng(42)
    y, x = np.mgrid[-3:3:96j, -4:4:160j]
    matrix = (np.sin(2.5 * x + y) * np.cos(1.6 * y) * np.exp(-(x * x + y * y) / 11)
              + 0.035 * rng.normal(size=x.shape)).astype(np.float32)
    timeline = np.linspace(0, 12, 800)
    signal = (np.sin(5 * timeline) * np.exp(-0.18 * timeline) + 0.15 * np.sin(18 * timeline)).astype(np.float32)
    centers = rng.normal(0, 1.7, (4, 24))
    embeddings = (centers[np.arange(360) % 4] + rng.normal(0, 0.62, (360, 24))).astype(np.float32)
    maps = np.stack([np.roll(matrix, 12 * index, axis=1) for index in range(4)])
    baseline = {
        "encoder.projection.weight": matrix,
        "encoder.projection.bias": np.zeros(matrix.shape[0], dtype=np.float32),
        "features.layer_stack": maps,
        "features.signal": signal,
        "text.embeddings": embeddings,
        "training.steps": np.array(1200),
    }
    tuned = dict(baseline)
    tuned["encoder.projection.weight"] = matrix + rng.normal(0, 0.08, matrix.shape).astype(np.float32)
    tuned["encoder.projection.bias"] = rng.normal(0, 0.025, matrix.shape[0]).astype(np.float32)
    tuned["training.steps"] = np.array(1800)
    tuned["head.classifier.weight"] = rng.normal(0, 0.16, (12, 24)).astype(np.float32)
    return tuple(
        TensorFile(None, name, "Built-in example", {key: array_tensor(key, value) for key, value in arrays.items()},
                   {"Dataset": "Synthetic examples generated locally", "Checkpoint": name})
        for name, arrays in (("Baseline", baseline), ("Fine-tuned", tuned))
    )
