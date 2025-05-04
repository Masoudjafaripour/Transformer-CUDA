from .ops import matmul_kernel, relu_kernel

def feed_forward(x, W1, b1, W2, b2):
    """Position-wise FFN"""
    # First linear + ReLU
    hidden = matmul(x, W1) + b1
    hidden = relu(hidden)
    
    # Second linear
    return matmul(hidden, W2) + b2
