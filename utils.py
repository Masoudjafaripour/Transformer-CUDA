import numpy as np

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe.astype(np.float32)

def init_weights(d_in, d_out):
    return np.random.randn(d_in, d_out).astype(np.float32) * 0.02

def init_bias(dim):
    return np.zeros(dim).astype(np.float32)
