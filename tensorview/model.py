"""Shared data model and bounded reads for tensor sources."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from math import prod
from pathlib import Path

import numpy as np

Index = tuple[int | slice, ...]
CHUNK_ELEMENTS = 262_144
NUMERIC_KINDS = frozenset("buifc")


class TensorViewError(ValueError):
    """An input or operation the application cannot handle."""


@dataclass(frozen=True, slots=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    reader: Callable[[Index], np.ndarray] = field(repr=False, compare=False)
    stored_dtype: str = ""
    stored_itemsize: int = 0

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * (self.stored_itemsize or self.dtype.itemsize)

    @property
    def dtype_name(self) -> str:
        return self.stored_dtype or str(self.dtype)

    def read(self, index: Index = ()) -> np.ndarray:
        return np.asarray(self.reader(index))

    def blocks(self, limit: int = CHUNK_ELEMENTS) -> Iterator[np.ndarray]:
        for index in block_indices(self.shape, limit):
            yield self.read(index).reshape(-1)

    def preview(self, count: int = 1024) -> np.ndarray:
        if count <= 0 or self.size == 0:
            return np.empty(0, dtype=self.dtype)
        remaining = min(count, self.size)
        parts = []
        for block in self.blocks(remaining):
            parts.append(block[:remaining])
            remaining -= parts[-1].size
            if remaining == 0:
                break
        return np.concatenate(parts)


@dataclass(slots=True)
class TensorFile:
    path: Path | None
    label: str
    format: str
    tensors: dict[str, Tensor]
    metadata: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(tensor.size for tensor in self.tensors.values())

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())


def block_indices(shape: tuple[int, ...], limit: int = CHUNK_ELEMENTS) -> Iterator[Index]:
    """Partition the array into contiguous logical C-order blocks."""
    if limit < 1:
        raise ValueError("Block limit must be positive.")
    if 0 in shape:
        return
    if not shape:
        yield ()
        return
    suffix = 1
    for axis in range(len(shape) - 1, -1, -1):
        if suffix * shape[axis] > limit:
            break
        suffix *= shape[axis]
    else:
        yield (slice(None),) * len(shape)
        return
    step = limit // suffix
    trailing = (slice(None),) * (len(shape) - axis - 1)
    prefix_shape = shape[:axis]
    for position in range(prod(prefix_shape)):
        coordinates = []
        for length in reversed(prefix_shape):
            position, coordinate = divmod(position, length)
            coordinates.append(coordinate)
        prefix = tuple(reversed(coordinates))
        for start in range(0, shape[axis], step):
            yield (*prefix, slice(start, min(start + step, shape[axis])), *trailing)


def array_tensor(name: str, array: np.ndarray) -> Tensor:
    return Tensor(name, array.shape, array.dtype, array.__getitem__)


def numeric_dtype(dtype: np.dtype) -> bool:
    return dtype.kind in NUMERIC_KINDS and (
        dtype.itemsize <= (16 if dtype.kind == "c" else 8)
    )


def unique_name(tensors: dict[str, Tensor], name: str) -> str:
    if name not in tensors:
        return name
    return next(
        candidate for number in range(2, len(tensors) + 2)
        if (candidate := f"{name} [{number}]") not in tensors
    )


def short_metadata(value: object) -> object:
    if isinstance(value, str):
        return value[:2048]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [short_metadata(item) for item in islice(value, 16)]
    return str(value)[:2048]
