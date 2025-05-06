import numpy as np
from numba import cuda
from .attention import CUDAAttention

class CUDAFFN:
    def __init__(self):
        self.attention = CUDAAttention()
        self.TILE_SIZE = 16

    @staticmethod
    @cuda.jit
    def _relu_kernel(X, out):
        """CUDA kernel for ReLU activation"""
        row, col = cuda.grid(2)
        if row < X.shape[0] and col < X.shape[1]:
            val = X[row, col]
            out[row, col] = max(0.0, val)

    def apply_relu(self, X):
        """Apply ReLU activation on GPU"""
        seq_len, d_ff = X.shape
        relu_out = np.zeros_like(X)

        # Configure grid
        threads = (self.TILE_SIZE, self.TILE_SIZE)
        blocks = ((seq_len + threads[0] - 1) // threads[0],
                 (d_ff + threads[1] - 1) // threads[1])

        # Execute kernel
        d_X = cuda.to_device(X)
        d_out = cuda.to_device(relu_out)
        self._relu_kernel[blocks, threads](d_X, d_out)
        return d_out.copy_to_host()

    def forward(self, X, W1, b1, W2, b2):
        """Forward pass through the feed-forward network"""
        # First linear projection
        intermediate = self.attention.matrix_multiply(X, W1) + b1

        # ReLU activation
        relu_out = self.apply_relu(intermediate)

        # Second linear projection
        output = self.attention.matrix_multiply(relu_out, W2) + b2
        return output
