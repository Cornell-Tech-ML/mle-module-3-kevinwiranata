import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numba
import numpy as np

import minitorch

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    print(backend)
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup

    if numba.cuda.is_available():
        print("Running GPU")
        run_matmul(GPUBackend)
    else:
        print("Running CPU")
        run_matmul(FastTensorBackend)

    ntrials = 3
    times = {}
    for size in [32, 64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

    plt.plot(times.keys(), [t["fast"] for t in times.values()], label="CPU")
    plt.plot(times.keys(), [t["gpu"] for t in times.values()], label="GPU")
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title("Performance of _tensor_matrix_multiply")
    plt.grid()
    plt.legend()
    plt.show()
