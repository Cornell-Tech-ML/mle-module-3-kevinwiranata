import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import minitorch
import minitorch.fast_ops
from minitorch import TensorBackend

# Define problem sizes
problem_sizes = [10, 100, 250, 500, 1000, 2500, 5000]

# Initialize list to store times
times = []

# For each problem size
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
shared: Dict[str, TensorBackend] = {"fast": FastTensorBackend}
shared["cuda"] = minitorch.TensorBackend(minitorch.CudaOps)

for N in problem_sizes:
    # Initialize NxN tensor
    tensor = np.random.rand(N, N)
    # Assuming `tensor` is a numpy ndarray
    # tensor = minitorch.Tensor(tensor, backend=shared["cuda"])

    # Start timer
    start_time = time.time()

    # Perform tensor matrix multiplication
    tensor @ tensor

    # Stop timer and record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

# Plot problem sizes against times
plt.plot(problem_sizes, times)
plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("Performance of _tensor_matrix_multiply")
plt.grid()
plt.legend()
plt.show()
print(problem_sizes, times)
