# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``ops.swap_blocks_batch`` C++ (cuMemcpyBatchAsync) path."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def _addrs(buffers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64)


def _run_batch(sizes: list[int]) -> None:
    src = [torch.randint(256, (s,), dtype=torch.uint8, device="cuda") for s in sizes]
    dst = [torch.zeros_like(s) for s in src]
    ops.swap_blocks_batch(
        _addrs(src), _addrs(dst), torch.tensor(sizes, dtype=torch.int64)
    )
    torch.accelerator.synchronize()
    for s, d in zip(src, dst):
        assert torch.equal(d, s)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="swap_blocks_batch requires CUDA"
)
def test_swap_blocks_batch_default_stream():
    # cuMemcpyBatchAsync rejects the legacy default stream; the op must fall
    # back to per-copy transfers instead of raising.
    _run_batch([8, 4096, 8192])


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="swap_blocks_batch requires CUDA"
)
def test_swap_blocks_batch_dedicated_stream():
    # A dedicated non-default stream exercises the cuMemcpyBatchAsync fast path.
    with torch.cuda.stream(torch.cuda.Stream()):
        _run_batch([8, 4096, 8192])


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("dedicated_stream", [False, True])
def test_copy_tensors_to_cpu_preserves_ragged_shapes_and_dtypes(dedicated_stream):
    """Ragged outputs share pinned storage without padding changing values."""
    from contextlib import nullcontext

    sources = [
        torch.arange(15, device="cuda", dtype=torch.int8),
        torch.randn(3, 5, device="cuda", dtype=torch.bfloat16),
        torch.randn(4, 3, device="cuda").t(),
        torch.tensor(3.5, device="cuda", dtype=torch.float64),
        torch.empty(0, 3, device="cuda"),
        torch.randn(1, device="cuda").expand(7),
    ]
    main_stream = torch.cuda.current_stream()
    context = (
        torch.cuda.stream(torch.cuda.Stream()) if dedicated_stream else nullcontext()
    )
    with context:
        torch.cuda.current_stream().wait_stream(main_stream)
        outputs = ops.copy_tensors_to_cpu(sources)
        torch.cuda.current_stream().synchronize()
    assert len({t.untyped_storage().data_ptr() for t in outputs}) == 1
    for source, output in zip(sources, outputs, strict=True):
        assert output.is_pinned()
        assert output.is_contiguous()
        assert output.dtype == source.dtype
        torch.testing.assert_close(output, source.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("rows", [[0, 1, 2, 3], [3, 1, 1, 0], []])
@pytest.mark.parametrize("layout", ["dense", "padded_rows", "transposed", "empty"])
def test_copy_rows_to_cpu_preserves_selection(rows, layout):
    source = torch.randn(4, 3, 5, device="cuda", dtype=torch.bfloat16)
    if layout == "padded_rows":
        source = source[:, :2]
    elif layout == "transposed":
        source = source.transpose(1, 2)
    elif layout == "empty":
        source = source[:, :0]
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(main_stream)
        outputs = ops.copy_rows_to_cpu(source, rows)
        copy_stream.synchronize()
    assert len(outputs) == len(rows)
    for row, output in zip(rows, outputs, strict=True):
        torch.testing.assert_close(output, source[row].cpu(), rtol=0, atol=0)
    if outputs:
        assert len({t.untyped_storage().data_ptr() for t in outputs}) == 1


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA/HIP")
def test_copy_rows_to_cpu_rejects_invalid_indices_before_copying():
    source = torch.randn(4, 8, device="cuda")
    with pytest.raises(RuntimeError, match="row out of range"):
        ops.copy_rows_to_cpu(source, [0, 4])
    with pytest.raises(RuntimeError, match="row out of range"):
        ops.copy_rows_to_cpu(source, [-1])
    with pytest.raises(RuntimeError, match="row dimension"):
        ops.copy_rows_to_cpu(source[0, 0], [0])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("as_rows", [False, True])
def test_pending_host_copies_complete_when_last_output_is_dropped(as_rows):
    """Dropping an unfinished output waits before releasing its allocations."""
    source = torch.randn(8, 1024, device="cuda")
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    before_copy = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(main_stream)
        torch.cuda._sleep(100_000_000)
        before_copy.record(copy_stream)
        outputs = (
            ops.copy_rows_to_cpu(source, list(range(8)))
            if as_rows
            else ops.copy_tensors_to_cpu([source])
        )
    del source, outputs
    assert before_copy.query()


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA/HIP")
def test_completed_host_copy_releases_source_without_another_submission():
    """Release GPU sources at the existing output completion boundary."""
    torch.accelerator.synchronize()
    baseline = torch.accelerator.memory_allocated()
    source = torch.ones(1024, 1024, device="cuda")
    outputs = ops.copy_tensors_to_cpu([source])
    del source
    torch.cuda.current_stream().synchronize()
    ops.release_completed_host_copies()
    assert torch.accelerator.memory_allocated() <= baseline
    assert outputs[0].shape == (1024, 1024)
    assert torch.all(outputs[0] == 1)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("strided", [False, True])
def test_pending_copy_does_not_retain_python_row_views(strided):
    """In-flight ownership must retain storage rather than Python row objects."""
    import weakref

    source = torch.randn(8, 32, device="cuda")
    if strided:
        source = source[:, ::2]
    rows = list(source.unbind())
    references = [weakref.ref(row) for row in rows]
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(main_stream)
        torch.cuda._sleep(100_000_000)
        outputs = ops.copy_tensors_to_cpu(rows)
    del rows
    assert all(reference() is None for reference in references)
    copy_stream.synchronize()
    torch.testing.assert_close(torch.stack(outputs), source.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [False, True])
def test_copy_helpers_allow_graph_breaks_in_compiled_callers(rows):
    def pipeline(tensor):
        source = tensor + 1
        output = (
            ops.copy_rows_to_cpu(source, [0])[0]
            if rows
            else ops.copy_tensors_to_cpu([source])[0]
        )
        return output + 2

    source = torch.randn(2, 3)
    output = torch.compile(pipeline, backend="eager")(source)
    torch.testing.assert_close(output, (source[0] if rows else source) + 3)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA/HIP")
def test_copy_tensors_to_cpu_preserves_mixed_cpu_and_cuda_outputs():
    cpu = torch.arange(2)
    gpu = torch.arange(3, device="cuda")
    outputs = ops.copy_tensors_to_cpu([gpu, cpu])
    del gpu
    torch.accelerator.synchronize()
    ops.release_completed_host_copies()
    assert outputs[1] is cpu
    assert outputs[0].is_pinned()
    torch.testing.assert_close(outputs[0], torch.arange(3), rtol=0, atol=0)
