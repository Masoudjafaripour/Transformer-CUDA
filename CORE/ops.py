import numpy as np
from numba import cuda
import math

def gpu_matmul(A, B):
    """GPU-accelerated matrix multiplication"""
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"Incompatible shapes: {A.shape}, {B.shape}")

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    # Copy matrices to GPU
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array((m, n), dtype=np.float32)

    # Configure CUDA grid
    TILE_SIZE = 16
    threads_per_block = (TILE_SIZE, TILE_SIZE)
    blocks_per_grid_x = (n + TILE_SIZE - 1) // TILE_SIZE
    blocks_per_grid_y = (m + TILE_SIZE - 1) // TILE_SIZE
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    matmul_kernel[blocks_per_grid, threads_per_block](dA, dB, dC)
    return dC.copy_to_host()

@cuda.jit
def matmul_kernel(A, B, C):
    """CUDA kernel for matrix multiplication"""
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for i in range(A.shape[1]):
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

@cuda.jit
def softmax_kernel(scores, result, seq_length):
    """CUDA kernel for softmax operation"""
    row = cuda.grid(1)
    if row < seq_length:
        # Find max value for numerical stability
        max_val = -float('inf')
        for i in range(seq_length):
            max_val = max(max_val, scores[row, i])

        # Compute exp and sum
        sum_exp = 0.0
        for i in range(seq_length):
            val = math.exp(scores[row, i] - max_val)
            result[row, i] = val
            sum_exp += val

        # Normalize
        for i in range(seq_length):
            result[row, i] /= sum_exp

@cuda.jit
def layer_norm_kernel(x, weight, bias, output, eps=1e-5):
    """CUDA kernel for layer normalization"""
    row = cuda.grid(1)
    if row < x.shape[0]:
        # Compute mean
        mean = 0.0
        for i in range(x.shape[1]):
            mean += x[row, i]
        mean /= x.shape[1]
        
        # Compute variance
        var = 0.0
        for i in range(x.shape[1]):
            var += (x[row, i] - mean)**2
        var = math.sqrt(var / x.shape[1] + eps)
        
        # Normalize and scale
        for i in range(x.shape[1]):
            output[row, i] = weight[i] * ((x[row, i] - mean) / var) + bias[i]

def layer_norm(x, weight, bias, eps=1e-5):
    """GPU-accelerated layer normalization"""
    output = np.zeros_like(x)
    d_x = cuda.to_device(x)
    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)
    d_output = cuda.to_device(output)
    
    threads_per_block = 32
    blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
    
    layer_norm_kernel[blocks_per_grid, threads_per_block](d_x, d_weight, d_bias, d_output, eps)
    return d_output.copy_to_host()
