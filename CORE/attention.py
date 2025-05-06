from .ops import matmul_kernel, softmax_kernel
import numpy as np
import math
from numba import cuda

def attention(q, k, v, mask=None):
    """Scaled dot-product attention"""
    d_k = q.shape[-1]
    scores = matmul(q, k.T) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn = softmax(scores)
    return matmul(attn, v), attn

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

def self_attention_cuda(X, W_Q, W_K, W_V):
    """CUDA-accelerated self-attention computation"""
    seq_length, d_model = X.shape
    d_k = W_Q.shape[1]
    d_v = W_V.shape[1]

    # Compute Q, K, V on GPU
    Q = gpu_matmul(X, W_Q)  # (seq_len, d_k)
    K = gpu_matmul(X, W_K)  # (seq_len, d_k)
    V = gpu_matmul(X, W_V)  # (seq_len, d_v)

    # Compute attention scores
    K_T = K.T.astype(np.float32)
    scores = gpu_matmul(Q, K_T) / math.sqrt(d_k)

    # Apply softmax on GPU
    attention_weights = np.zeros((seq_length, seq_length), dtype=np.float32)
    d_scores = cuda.to_device(scores)
    d_attention_weights = cuda.to_device(attention_weights)

    threads_per_block = 32
    blocks_per_grid = (seq_length + threads_per_block - 1) // threads_per_block
    softmax_kernel[blocks_per_grid, threads_per_block](d_scores, d_attention_weights, seq_length)
    attention_weights = d_attention_weights.copy_to_host()

    # Compute output
    output = gpu_matmul(attention_weights, V)
    return output, attention_weights

def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads):
    """CUDA-accelerated multi-head attention"""
    seq_length, d_model = X.shape

    # Calculate dimensions for each head
    d_k = W_Q.shape[1] // num_heads
    d_v = W_V.shape[1] // num_heads

    outputs = []
    all_attention_weights = []

    # Process each attention head
    for i in range(num_heads):
        # Slice weights for this head
        W_Q_i = W_Q[:, i*d_k:(i+1)*d_k]
        W_K_i = W_K[:, i*d_k:(i+1)*d_k]
        W_V_i = W_V[:, i*d_v:(i+1)*d_v]

        # Compute attention for this head
        out_i, attn_i = self_attention_cuda(X, W_Q_i, W_K_i, W_V_i)
        
        outputs.append(out_i)
        all_attention_weights.append(attn_i)

    # Concatenate all heads
    concat = np.concatenate(outputs, axis=-1).astype(np.float32)

    # Final linear projection
    final_output = gpu_matmul(concat, W_O)

    return final_output, all_attention_weights
