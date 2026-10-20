# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare per-output D2H copies with packed native copies.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_copy_to_cpu.py --contend-gil

Measures CPU enqueue and output destruction separately. GPU completion is
synchronized outside both intervals. Run with --independent to prevent merging
adjacent source rows into a single transfer.

"""

import argparse
import statistics
import threading
import time

import torch

from vllm import _custom_ops as ops


def measure(copy, iterations):
    enqueue, destroy = [], []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        outputs = copy()
        enqueue.append((time.perf_counter_ns() - start) / 1000)
        torch.cuda.current_stream().synchronize()
        start = time.perf_counter_ns()
        del outputs
        destroy.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(enqueue), statistics.median(destroy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--contend-gil", action="store_true")
    args = parser.parse_args()
    stop = threading.Event()

    def contend():
        while not stop.is_set():
            pass

    print("batch  method       enqueue_us  destroy_us")
    for batch in args.batch_sizes:
        sources = (
            [torch.randn(args.width, device="cuda") for _ in range(batch)]
            if args.independent
            else list(torch.randn(batch, args.width, device="cuda").unbind())
        )
        torch.accelerator.synchronize()
        methods = {
            "per-output": lambda sources=sources: [
                t.to("cpu", non_blocking=True) for t in sources
            ],
            "packed": lambda sources=sources: ops.copy_tensors_to_cpu(sources),
        }
        with torch.cuda.stream(torch.cuda.Stream()):
            for copy in methods.values():
                measure(copy, 3)
            worker = threading.Thread(target=contend) if args.contend_gil else None
            if worker is not None:
                worker.start()
            try:
                for name, copy in methods.items():
                    enqueue, destroy = measure(copy, args.iterations)
                    print(f"{batch:5d}  {name:12s} {enqueue:10.1f}  {destroy:10.1f}")
            finally:
                stop.set()
                if worker is not None:
                    worker.join()
                stop.clear()


if __name__ == "__main__":
    main()
