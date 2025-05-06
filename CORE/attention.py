import numpy as np
from numba import cuda
import math

class CUDAAttention:
    def __init__(self):
        self.TILE_SIZE = 16
        self.THREADS_PER_BLOCK = 32

    @staticmethod
    @cuda.jit
    def _matmul_kernel(A, B, C):
        """CUDA kernel for matrix multiplication"""
        row, col = cuda.grid(2)
        if row < C.shape[0] and col < C.shape[1]:
            tmp = 0.0
            for i in range(A.shape[1]):
                tmp += A[row, i] * B[i, col]
            C[row, col] = tmp

    @staticmethod
    @cuda.jit
    def _softmax_kernel(scores, result, seq_length):
        """CUDA kernel for softmax operation"""
        row = cuda.grid(1)
        if row < seq_length:
            # Find max for numerical stability
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

    def matrix_multiply(self, A, B):
        """GPU-accelerated matrix multiplication"""
        m, k = A.shape
        k2, n = B.shape
        if k != k2:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape}, {B.shape}")

        # Prepare data
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.device_array((m, n), dtype=np.float32)

        # Configure grid
        threads = (self.TILE_SIZE, self.TILE_SIZE)
        blocks_x = (n + self.TILE_SIZE - 1) // self.TILE_SIZE
        blocks_y = (m + self.TILE_SIZE - 1) // self.TILE_SIZE
        blocks = (blocks_x, blocks_y)

        # Execute kernel
        self._matmul_kernel[blocks, threads](dA, dB, dC)
        return dC.copy_to_host()

    def compute_attention(self, X, W_Q, W_K, W_V):
        """Compute self-attention with CUDA acceleration"""
        seq_length, d_model = X.shape
        d_k = W_Q.shape[1]
        d_v = W_V.shape[1]

        # Compute Q, K, V matrices
        Q = self.matrix_multiply(X, W_Q)
        K = self.matrix_multiply(X, W_K)
        V = self.matrix_multiply(X, W_V)

        # Compute attention scores
        K_T = K.T.astype(np.float32)
        scores = self.matrix_multiply(Q, K_T) / math.sqrt(d_k)

        # Apply softmax
        attention_weights = np.zeros((seq_length, seq_length), dtype=np.float32)
        d_scores = cuda.to_device(scores)
        d_weights = cuda.to_device(attention_weights)

        blocks = (seq_length + self.THREADS_PER_BLOCK - 1) // self.THREADS_PER_BLOCK
        self._softmax_kernel[blocks, self.THREADS_PER_BLOCK](d_scores, d_weights, seq_length)
        attention_weights = d_weights.copy_to_host()

        # Compute final output
        output = self.matrix_multiply(attention_weights, V)
        return output, attention_weights

    def multi_head_attention(self, X, W_Q, W_K, W_V, W_O, num_heads):
        """Compute multi-head attention with CUDA acceleration"""
        seq_length, d_model = X.shape
        d_k = W_Q.shape[1] // num_heads
        d_v = W_V.shape[1] // num_heads

        outputs = []
        attention_weights = []

        # Process each attention head
        for i in range(num_heads):
            # Extract weights for this head
            W_Q_i = W_Q[:, i*d_k:(i+1)*d_k]
            W_K_i = W_K[:, i*d_k:(i+1)*d_k]
            W_V_i = W_V[:, i*d_v:(i+1)*d_v]

            # Compute attention for this head
            out_i, attn_i = self.compute_attention(X, W_Q_i, W_K_i, W_V_i)
            outputs.append(out_i)
            attention_weights.append(attn_i)

        # Combine heads
        concat = np.concatenate(outputs, axis=-1).astype(np.float32)
        final_output = self.matrix_multiply(concat, W_O)

        return final_output, attention_weights
