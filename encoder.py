from core.attention import multi_head_attention
from core.ffn import feed_forward
from core.ops import layer_norm
from utils import positional_encoding

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.W_q = init_weights(d_model, d_model)
        self.W_k = init_weights(d_model, d_model)
        self.W_v = init_weights(d_model, d_model)
        self.W_o = init_weights(d_model, d_model)
        
        self.W1 = init_weights(d_model, d_ff)
        self.W2 = init_weights(d_ff, d_model)
        self.b1 = init_bias(d_ff)
        self.b2 = init_bias(d_model)
        
        self.num_heads = num_heads

    def __call__(self, x):
        # Add positional encoding
        x = x + positional_encoding(x.shape[0], x.shape[1])
        
        # Multi-head attention
        attn_out, _ = multi_head_attention(
            x, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads)
        
        # Add & Norm
        x = layer_norm(x + attn_out)
        
        # FFN
        ffn_out = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        
        # Final Add & Norm
        return layer_norm(x + ffn_out)
