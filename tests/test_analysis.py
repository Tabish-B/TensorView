import numpy as np
import pytest

from tensorview.analysis import compare, compare_tensor, demo_files, plot_slice, project, statistics, validate
from tensorview.model import array_tensor, block_indices


def test_blocks_and_preview_preserve_logical_order():
    data = np.arange(8 * 13 * 9).reshape(8, 13, 9).transpose(2, 0, 1)
    tensor = array_tensor("noncontiguous", data)
    chunks = list(tensor.blocks(limit=17))
    assert max(map(len, chunks)) <= 17
    np.testing.assert_array_equal(np.concatenate(chunks), data.reshape(-1))
    np.testing.assert_array_equal(tensor.preview(45), data.reshape(-1)[:45])
    assert list(block_indices((7, 0, 2))) == []
    assert list(block_indices(())) == [()]


def test_statistics_count_all_values_and_sample_large_tensors():
    data = np.linspace(-4, 6, 619_353)
    data[:4] = [0, np.nan, np.inf, -np.inf]
    stats = statistics(array_tensor("a", data))
    finite = data[np.isfinite(data)]
    assert (stats.count, stats.finite, stats.zeros, stats.nan, stats.positive_inf, stats.negative_inf) == (
        data.size, data.size - 3, 1, 1, 1, 1,
    )
    assert stats.sampled
    assert stats.sample.size <= 65_536
    assert stats.minimum == finite.min()
    assert stats.maximum == finite.max()
    assert stats.mean == pytest.approx(finite.mean())
    assert stats.std == pytest.approx(finite.std())
    assert stats.median == pytest.approx(np.median(finite), abs=1e-3)


def test_statistics_handles_empty_complex_and_extreme_values():
    empty = statistics(array_tensor("empty", np.empty((0, 2))))
    assert empty.count == 0 and empty.minimum is None
    stats = statistics(array_tensor("z", np.array([3 + 4j, 0, complex(2, np.inf), complex(np.nan, 3)])))
    assert stats.complex
    assert (stats.nan, stats.infinite, stats.finite, stats.zeros) == (1, 1, 2, 1)
    assert stats.mean == 2.5
    assert stats.median == 2.5
    extreme = statistics(array_tensor("extreme", np.array([-1e308, 1e308])))
    assert extreme.mean == 0
    assert extreme.std == pytest.approx(1e308)
    integers = statistics(array_tensor("wide", np.array([2**63, 2**63 + 1], dtype=np.uint64)))
    assert integers.maximum == 2**63 + 1


def test_comparison_statuses_and_finite_metrics():
    a = array_tensor("weight", np.array([0.0, 2.0, np.nan, np.inf]))
    b = array_tensor("weight", np.array([0.0, 3.0, np.nan, -np.inf]))
    result = compare_tensor("weight", a, b)
    assert result.status == "Modified"
    assert result.changed_values == 2
    assert result.finite_pairs == 2
    assert result.mean_absolute == pytest.approx(0.5)
    assert result.max_absolute == pytest.approx(1)
    assert result.cosine == pytest.approx(1)
    assert result.relative_l2 == pytest.approx(0.5)
    assert compare_tensor("weight", a, None).status == "Removed"
    assert compare_tensor("weight", None, b).status == "Added"
    assert compare_tensor("weight", a, array_tensor("b", np.empty((1, 4)))).status == "Shape changed"
    assert compare_tensor("weight", a, array_tensor("b", a.read().astype(np.float32))).status == "Dtype changed"


def test_comparison_distinguishes_nearby_wide_integers():
    a = array_tensor("a", np.array([2**63 + 1], dtype=np.uint64))
    b = array_tensor("b", np.array([2**63 + 2], dtype=np.uint64))
    result = compare_tensor("a", a, b)
    assert result.changed_values == 1
    assert result.max_absolute == pytest.approx(1)
    c = array_tensor("c", np.array([float(2**63)], dtype="float64"))
    mixed = compare_tensor("a", a, c)
    assert mixed.changed_values == 1
    assert mixed.max_absolute == pytest.approx(1)
    signed = array_tensor("signed", np.array([2**63 - 1, -2], dtype=np.int64))
    unsigned = array_tensor("unsigned", np.array([2**63, 3], dtype=np.uint64))
    assert compare_tensor("mixed", signed, unsigned).mean_absolute == pytest.approx(3)


def test_distance_uses_independent_scales_across_blocks():
    left = np.zeros(262_145)
    right = np.zeros_like(left)
    left[0] = 1e-200
    right[-1] = 1
    result = compare_tensor("a", array_tensor("a", left), array_tensor("a", right))
    assert result.cosine == 0
    assert result.relative_l2 == pytest.approx(1e200)
    assert result.max_absolute == pytest.approx(1)
    backward = compare_tensor("a", array_tensor("a", right), array_tensor("a", left))
    assert backward.relative_l2 == pytest.approx(1)
    assert backward.cosine == 0


def test_distance_with_subnormal_complex_values():
    a = array_tensor("a", np.array([complex(2e-310, 1e-310)]))
    b = array_tensor("b", np.array([complex(1e-310, 2e-310)]))
    result = compare_tensor("a", a, b)
    assert result.relative_l2 == pytest.approx((2 / 5) ** 0.5)
    assert result.cosine == pytest.approx(0.8)
    assert result.max_absolute == pytest.approx(2**0.5 * 1e-310, abs=0)


def test_distance_when_absolute_difference_overflows():
    a = array_tensor("a", np.array([1.7e308]))
    b = array_tensor("b", np.array([-1.7e308]))
    result = compare_tensor("a", a, b)
    assert result.max_absolute == np.inf
    assert result.relative_l2 == pytest.approx(2)
    assert result.cosine == pytest.approx(-1)


def test_validation_and_nested_example_comparison():
    baseline, candidate = demo_files()
    findings = validate(baseline)
    assert any(item.check == "All zero" for item in findings)
    results = {diff.name: diff for diff in compare(baseline, candidate)}
    assert results["features.signal"].status == "Identical"
    assert results["head.classifier.weight"].status == "Added"
    assert results["encoder.projection.weight"].changed_values > 0


def test_slice_axis_order_and_pca():
    data = np.arange(4 * 11 * 8).reshape((4, 11, 8))
    output, steps, lengths = plot_slice(array_tensor("x", data), (2, 1), {0: 2})
    np.testing.assert_array_equal(output, data[2].T)
    assert steps == (1, 1)
    assert lengths == (8, 11)
    pytest.importorskip("sklearn")
    baseline, _ = demo_files()
    projection, indices = project(baseline.tensors["text.embeddings"], "PCA")
    assert projection.shape == (360, 2)
    np.testing.assert_array_equal(indices, np.arange(360))
