import numpy as np
from numba import cuda
from .ops import gpu_matmul

@cuda.jit
def relu_kernel(X, out):
    """CUDA kernel for ReLU activation"""
    row, col = cuda.grid(2)
    if row < X.shape[0] and col < X.shape[1]:
        val = X[row, col]
        out[row, col] = max(0.0, val)

def feed_forward(X, W1, b1, W2, b2):
    """CUDA-accelerated feed-forward network"""
    # First linear projection
    intermediate = gpu_matmul(X, W1) + b1

    # ReLU activation
    seq_len, d_ff = intermediate.shape
    relu_out = np.zeros_like(intermediate)

    # Configure CUDA grid for ReLU
    threads = (16, 16)
    blocks = ((seq_len + threads[0] - 1) // threads[0],
              (d_ff + threads[1] - 1) // threads[1])

    # Apply ReLU on GPU
    d_intermediate = cuda.to_device(intermediate)
    d_relu_out = cuda.to_device(relu_out)
    relu_kernel[blocks, threads](d_intermediate, d_relu_out)
    relu_out = d_relu_out.copy_to_host()

    # Second linear projection
    output = gpu_matmul(relu_out, W2) + b2
    return output
