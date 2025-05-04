import numpy as np
from numba import cuda, float32

@cuda.jit
def matmul_kernel(A, B, C, TILE=16):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for i in range(A.shape[1]):
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

@cuda.jit
def softmax_kernel(x, output):
    row = cuda.grid(1)
    if row < x.shape[0]:
        max_val = -float('inf')
        for i in range(x.shape[1]):
            max_val = max(max_val, x[row, i])
        
        sum_exp = 0.0
        for i in range(x.shape[1]):
            val = math.exp(x[row, i] - max_val)
            output[row, i] = val
            sum_exp += val

        for i in range(x.shape[1]):
            output[row, i] /= sum_exp

@cuda.jit
def layer_norm_kernel(x, weight, bias, output, eps=1e-5):
    row = cuda.grid(1)
    if row < x.shape[0]:
        mean = 0.0
        var = 0.0
        
        # Compute mean and variance
        for i in range(x.shape[1]):
            mean += x[row, i]
        mean /= x.shape[1]
        
        for i in range(x.shape[1]):
            var += (x[row, i] - mean)**2
        var = math.sqrt(var / x.shape[1] + eps)
        
        # Normalize and scale
        for i in range(x.shape[1]):
            output[row, i] = weight[i] * ((x[row, i] - mean) / var) + bias[i]
