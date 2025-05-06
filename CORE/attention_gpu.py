import numpy as np
from numba import cuda, float32
import math

__all__ = ["gpu_matmul", "gpu_matmul_optimized", "self_attention_cuda"]

@cuda.jit
def matmul_kernel(A, B, C):
    #this is the basic GPU matrix multiplication kernel (no shared memory).
    #A: (m, k)
    #B: (k, n)
    #C: (m, n)
    #row = row = blockIdx.y * blockDim.y + threadIdx.y
    #col = blockIdx.x * blockDim.x + threadIdx.x

    col, row = cuda.grid(2) #note: 2 is chosen here because we want each thread to correspond to a 2-D coordinate (row, col)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        
        # Dot product along k dimension (matrix multiplication)
        for i in range(A.shape[1]):
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

def gpu_matmul(A, B):
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        #to multiply matrices, the inner dimensions must match (# of columns in A must equal # of rows in B)
        raise ValueError(f"Incompatible shapes: {A.shape}, {B.shape}")

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    #copying matrices from the CPU to the GPU 
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)

    #creating an empty matrix on the GPU where the answer will go 
    dC = cuda.device_array((m, n), dtype=np.float32)

    TILE_SIZE = 16 #(16 x 16 is a CUDA best practice)
    threads_per_block = (TILE_SIZE, TILE_SIZE)

    # x dimension = n, y dimension = m
    #calculating how many blocks will be needed to cover the entire output matrix (will round up)
    blocks_per_grid_x = (n + TILE_SIZE - 1) // TILE_SIZE
    blocks_per_grid_y = (m + TILE_SIZE - 1) // TILE_SIZE
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    #launching the matmul kernel 
    matmul_kernel[blocks_per_grid, threads_per_block](dA, dB, dC)
    return dC.copy_to_host()

@cuda.jit
def softmax_kernel(scores, result, seq_length):
    #scores: 2-D matrix with attention scores for each token
    #result: matrix that stores the attention scores
    #seq_length: how many columns and rows the matrix has [sqaure matrix]

    #Assigning a thread to each row of the attention scores matrix
    row = cuda.grid(1)

    #check to make sure no thread goes out of bounds
    if row < seq_length:
        #this is the lowest number max_val can be assigned to (negative infinite), so any number will be greater
        max_val = -float('inf')
        for i in range(seq_length):
            max_val = max(max_val, scores[row, i])

        sum_exp = 0.0
        for i in range(seq_length):
            #creating a numerically stable softmax (prevents overflow errors from occuring)
            val = math.exp(scores[row, i] - max_val)
            result[row, i] = val
            sum_exp += val

        #normalizing the values to get probabilities
        for i in range(seq_length):
            result[row, i] /= sum_exp

def self_attention_cuda(X, W_Q, W_K, W_V):
    seq_length, d_model = X.shape
    d_k = W_Q.shape[1]
    d_v = W_V.shape[1]

    #computing Q, K, V on GPU 
    Q = gpu_matmul(X, W_Q)  # (seq_len, d_k)
    K = gpu_matmul(X, W_K)  # (seq_len, d_k)
    V = gpu_matmul(X, W_V)  # (seq_len, d_v)

    #computing attention scores
    K_T = K.T.astype(np.float32)
    scores = gpu_matmul(Q, K_T) / math.sqrt(d_k)

    #sotmaxing the scores on a GPU 
    attention_weights = np.zeros((seq_length, seq_length), dtype=np.float32)
    d_scores = cuda.to_device(scores)
    d_attention_weights = cuda.to_device(attention_weights)

    threads_per_block = 32
    blocks_per_grid = (seq_length + threads_per_block - 1) // threads_per_block
    softmax_kernel[blocks_per_grid, threads_per_block](d_scores, d_attention_weights, seq_length)
    attention_weights = d_attention_weights.copy_to_host()

    #computing output 
    output = gpu_matmul(attention_weights, V)
    return output, attention_weights 