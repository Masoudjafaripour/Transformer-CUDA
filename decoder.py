from core.attention import multi_head_attention
from core.ffn import feed_forward
from core.ops import layer_norm
from utils import positional_encoding, init_weights, init_bias

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        # Masked Self-Attention parameters
        self.W_q_self = init_weights(d_model, d_model)
        self.W_k_self = init_weights(d_model, d_model)
        self.W_v_self = init_weights(d_model, d_model)
        self.W_o_self = init_weights(d_model, d_model)

        # Cross-Attention parameters (attending to encoder output)
        self.W_q_cross = init_weights(d_model, d_model)
        self.W_k_cross = init_weights(d_model, d_model)
        self.W_v_cross = init_weights(d_model, d_model)
        self.W_o_cross = init_weights(d_model, d_model)

        # Feedforward Network
        self.W1 = init_weights(d_model, d_ff)
        self.W2 = init_weights(d_ff, d_model)
        self.b1 = init_bias(d_ff)
        self.b2 = init_bias(d_model)

        self.num_heads = num_heads

    def __call__(self, x, encoder_output):
        # Add positional encoding to decoder input
        x = x + positional_encoding(x.shape[0], x.shape[1])

        # Masked Multi-head Self-Attention (masking to be implemented later)
        self_attn_out, _ = multi_head_attention(
            x, self.W_q_self, self.W_k_self, self.W_v_self, self.W_o_self, self.num_heads)
        x = layer_norm(x + self_attn_out)

        # Cross-Attention over encoder output
        cross_attn_out, _ = multi_head_attention(
            x, self.W_q_cross, encoder_output @ self.W_k_cross, encoder_output @ self.W_v_cross, self.W_o_cross, self.num_heads)
        x = layer_norm(x + cross_attn_out)

        # Feedforward Network
        ffn_out = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        return layer_norm(x + ffn_out)
