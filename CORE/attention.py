from .ops import matmul_kernel, softmax_kernel
import numpy as np
import math

def attention(q, k, v, mask=None):
    """Scaled dot-product attention"""
    d_k = q.shape[-1]
    scores = matmul(q, k.T) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn = softmax(scores)
    return matmul(attn, v), attn

def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads):
    """Parallel attention heads"""
    batch, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # Split into multiple heads
    q = matmul(x, W_q).view(batch, seq_len, num_heads, d_k)
    k = matmul(x, W_k).view(batch, seq_len, num_heads, d_k) 
    v = matmul(x, W_v).view(batch, seq_len, num_heads, d_k)
    
    # Transpose for attention computation
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, d_k]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Compute attention
    scores, attn = attention(q, k, v)
    
    # Concatenate heads
    concat = scores.transpose(1, 2).contiguous() \
             .view(batch, seq_len, -1)
    
    # Project back to original dimension
    return matmul(concat, W_o), attn
