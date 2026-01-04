"""Transformer Decoder modules."""

import math

import torch
import torch.nn as nn

from transformer.nn.attention import MultiHeadAttention
from transformer.nn.embedding import PositionalEncoding
from transformer.nn.feedforward import FeedForward


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer from "Attention Is All You Need".

    Each layer consists of:
    1. Masked Multi-Head Self-Attention (with residual connection and layer norm)
    2. Multi-Head Cross-Attention over encoder output (with residual connection and layer norm)
    3. Position-wise Feed-Forward Network (with residual connection and layer norm)

    Args:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        d_ff: Feed-forward inner dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # Cross-attention (attends to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attn_mask: Mask for self-attention (look-ahead mask)
            cross_attn_mask: Mask for cross-attention (padding mask for encoder)

        Returns:
            output: Decoder output of shape (batch_size, tgt_seq_len, d_model)
            self_attention_weights: Self-attention weights
            cross_attention_weights: Cross-attention weights
        """
        # Masked self-attention with residual connection and layer norm
        self_attn_output, self_attention_weights = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-attention with residual connection and layer norm
        # Query from decoder, Key and Value from encoder
        cross_attn_output, cross_attention_weights = self.cross_attn(
            x, encoder_output, encoder_output, cross_attn_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, self_attention_weights, cross_attention_weights


class Decoder(nn.Module):
    """
    Transformer Decoder from "Attention Is All You Need".

    Consists of:
    1. Output Embedding
    2. Positional Encoding
    3. Stack of N Decoder Layers

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension (embedding size)
        n_layers: Number of decoder layers (default: 6)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward inner dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_len: Maximum sequence length for positional encoding (default: 5000)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            x: Target token indices of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attn_mask: Look-ahead mask for self-attention
            cross_attn_mask: Padding mask for cross-attention

        Returns:
            output: Decoder output of shape (batch_size, tgt_seq_len, d_model)
            self_attention_weights: List of self-attention weights from each layer
            cross_attention_weights: List of cross-attention weights from each layer
        """
        # Embed tokens and scale by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through decoder layers
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, self_attn_mask, cross_attn_mask
            )
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)

        # Final layer normalization
        x = self.norm(x)

        return x, self_attn_weights_list, cross_attn_weights_list
