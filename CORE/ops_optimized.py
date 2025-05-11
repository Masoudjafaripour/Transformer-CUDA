import numpy as np
from numba import cuda, float32
import math

# -------------------------------
# Optimized Matrix Multiplication
# -------------------------------

TILE_SIZE = 16

@cuda.jit
def matmul_kernel_shared(A, B, C):
    sA = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    sB = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE_SIZE + ty
    col = cuda.blockIdx.x * TILE_SIZE + tx

    tmp = 0.0
    for t in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):
        if row < A.shape[0] and (t * TILE_SIZE + tx) < A.shape[1]:
            sA[ty, tx] = A[row, t * TILE_SIZE + tx]
        else:
            sA[ty, tx] = 0.0

        if col < B.shape[1] and (t * TILE_SIZE + ty) < B.shape[0]:
            sB[ty, tx] = B[t * TILE_SIZE + ty, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(TILE_SIZE):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

def gpu_matmul(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    m, k = A.shape
    k2, n = B.shape
    assert k == k2

    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array((m, n), dtype=np.float32)

    threads = (TILE_SIZE, TILE_SIZE)
    blocks = ((n + TILE_SIZE - 1) // TILE_SIZE, (m + TILE_SIZE - 1) // TILE_SIZE)

    matmul_kernel_shared[blocks, threads](dA, dB, dC)
    return dC.copy_to_host()


# -------------------
# Optimized Softmax
# -------------------

@cuda.jit
def softmax_kernel_parallel(scores, result):
    row = cuda.grid(1)
    if row >= scores.shape[0]:
        return

    # Shared memory for row softmax
    sdata = cuda.shared.array(512, dtype=float32)
    max_val = -1e20

    for j in range(scores.shape[1]):
        max_val = max(max_val, scores[row, j])

    sum_exp = 0.0
    for j in range(scores.shape[1]):
        val = math.exp(scores[row, j] - max_val)
        sdata[j] = val
        sum_exp += val

    for j in range(scores.shape[1]):
        result[row, j] = sdata[j] / sum_exp

def gpu_softmax(scores):
    scores = scores.astype(np.float32)
    result = np.zeros_like(scores)
    d_scores = cuda.to_device(scores)
    d_result = cuda.device_array_like(result)

    threads_per_block = 128
    blocks_per_grid = (scores.shape[0] + threads_per_block - 1) // threads_per_block

    softmax_kernel_parallel[blocks_per_grid, threads_per_block](d_scores, d_result)
    return d_result.copy_to_host()


# -------------------
# Optimized LayerNorm
# -------------------

@cuda.jit
def layer_norm_kernel_shared(x, weight, bias, output, eps=1e-5):
    row = cuda.grid(1)
    if row >= x.shape[0]:
        return

    d = x.shape[1]
    mean = 0.0
    var = 0.0

    for i in range(d):
        mean += x[row, i]
    mean /= d

    for i in range(d):
        diff = x[row, i] - mean
        var += diff * diff
    var = math.sqrt(var / d + eps)

    for i in range(d):
        output[row, i] = weight[i] * ((x[row, i] - mean) / var) + bias[i]

def layer_norm(x, weight, bias, eps=1e-5):
    x = x.astype(np.float32)
    weight = weight.astype(np.float32)
    bias = bias.astype(np.float32)
    output = np.zeros_like(x)

    d_x = cuda.to_device(x)
    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)
    d_output = cuda.device_array_like(output)

    threads_per_block = 128
    blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block

    layer_norm_kernel_shared[blocks_per_grid, threads_per_block](d_x, d_weight, d_bias, d_output, eps)
    return d_output.copy_to_host()
