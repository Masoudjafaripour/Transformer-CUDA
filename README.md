# CUDA-Accelerated Transformer (Encoder + Decoder)

🚀 A from-scratch implementation of a full Transformer model with CUDA acceleration — no PyTorch or TensorFlow required!

This project expands on a CUDA-based Transformer encoder by including a decoder, bringing it closer to a full sequence-to-sequence model like the original "Attention is All You Need" paper.

## 🔑 Key Features

* **CUDA Acceleration**: All core operations (matrix multiplication, softmax, layer normalization, ReLU) are implemented using custom CUDA kernels for GPU-accelerated computation.
* **Multi-Head Attention**: Implements parallel multi-head attention to capture diverse relationships across tokens.
* **Add & Norm Layers**: Residual connections and layer normalization improve training stability.
* **Feed-Forward Network**: Each token is passed through a 2-layer MLP using GPU matrix operations.
* **Positional Encoding**: Uses sinusoidal functions to inject token order into input embeddings.
* **Decoder Integration** *(New)*: Adds a masked self-attention layer and encoder-decoder cross-attention.

## 📦 Key Components

### Core Operations (`ops.py`)

* CUDA-accelerated matrix multiplication
* Numerically stable softmax
* Layer normalization kernel (per token)

### Attention Mechanism (`attention.py`)

* Multi-head self-attention
* Scaled dot-product attention
* Parallel attention heads

### Feed-Forward Network (`ffn.py`)

* Linear → ReLU → Linear pipeline
* CUDA ReLU kernel

### Encoder Block (`encoder.py`)

* Full encoder implementation
* Residual + LayerNorm
* Multi-head attention and FFN integration

### Decoder Block *(Coming Soon)*

* Masked multi-head self-attention
* Cross-attention over encoder outputs
* Additional residual and LayerNorm layers

### Utilities (`utils.py`)

* Sinusoidal positional encoding
* Weight and bias initialization

## 🔁 Data Flow (Encoder + Decoder)

```
Input → Positional Encoding
      → Encoder → Output (context)

Target Input → Positional Encoding
            → Masked Self-Attention
            → Cross Attention with Encoder Output
            → Feedforward + LayerNorm
            → Decoder Output
```

> A powerful, minimal framework to understand and extend the Transformer architecture at a low level using CUDA. Ideal for ML researchers, GPU enthusiasts, and systems programmers.

---

More features and decoder implementation in progress. Contributions welcome!

