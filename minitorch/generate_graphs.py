import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import minitorch
import minitorch.fast_ops
from minitorch import TensorBackend

# Define problem sizes
problem_sizes = [10, 50, 100, 150, 200, 250]

# Initialize list to store times_cpu
times_cpu = []
times_gpu = [4.220008850097656e-05, 9.083747863769531e-05, 0.00015306472778320312, 0.0005927085876464844, 0.0018389225006103516, 0.0010445117950439453]
# For each problem size
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
shared: Dict[str, TensorBackend] = {"fast": FastTensorBackend}
for N in problem_sizes:
    # Initialize NxN tensor
    tensor = np.random.rand(N, N)
    # Assuming `tensor` is a numpy ndarray
    # tensor = minitorch.Tensor(tensor, backend=shared["fast"])

    # Start timer
    start_time = time.time()

    # Perform tensor matrix multiplication
    tensor @ tensor

    # Stop timer and record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    times_cpu.append(elapsed_time)
    time.sleep(1)

# Plot problem sizes against times_cpu
plt.plot(problem_sizes, times_cpu, label="CPU")
plt.plot(problem_sizes, times_gpu, label="GPU")
plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("Performance of _tensor_matrix_multiply")
plt.grid()
plt.legend()
plt.show()
print(problem_sizes, times_cpu)
