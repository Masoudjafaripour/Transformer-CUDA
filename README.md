# CUDA-Accelerated Transformer Encoder Block
🚀 A from-scratch implementation of a Transformer encoder block with CUDA acceleration - no PyTorch/TensorFlow required!

## Key Features
- CUDA Acceleration: Utilizes CUDA kernels for matrix multiplication, softmax, and layer normalization, providing substantial speedups compared to CPU implementations.
- Multi-Head Attention: Implements multi-head attention mechanism to capture different relationships within the input sequence.
- Add & Norm: Includes residual connections and layer normalization for improved training stability and performance.
- Feed-Forward Network: A feed-forward network (FFN) implemented with CUDA to transform each token independently.
- Positional Encoding: Applies sinusoidal positional embeddings to the input to provide information about the position of tokens in the sequence.

## Key Components
- Core Operations (ops.py)

  - GPU-optimized matrix multiplication
  
  - Numerically stable softmax
  
  - Layer normalization kernel

- Attention Mechanism (attention.py)

  - Multi-head self-attention
  
  - Scaled dot-product computation
  
  - Parallel head processing

- Feed-Forward Network (ffn.py)

  - Position-wise transformations
  
  - ReLU activation
  
  - Linear projections

- Encoder Block (encoder.py)

  - Full encoder implementation
  
  - Residual connections
  
  - Sublayer integration

- Utilities (utils.py)

  - Sinusoidal positional encoding
  
  - Parameter initialization

# Data Flow
<img width="556" alt="image" src="https://github.com/user-attachments/assets/8dd2e665-fec5-4f4f-9773-5a50799dcb2f" />
