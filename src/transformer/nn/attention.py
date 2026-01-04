"""Multi-Head Attention module."""

import torch
import torch.nn as nn

from transformer.functional import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
    where head_i = Attention(Q * W_q_i, K * W_k_i, V * W_v_i)

    Args:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len) or
                  (batch_size, 1, seq_len, seq_len)

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections: (batch_size, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Reshape to (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        dropout_p = self.dropout_p if self.training else 0.0
        attn_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=dropout_p
        )

        # Reshape back: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.w_o(attn_output)

        return output, attention_weights
