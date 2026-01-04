"""Transformer Encoder modules."""

import math

import torch
import torch.nn as nn

from transformer.nn.attention import MultiHeadAttention
from transformer.nn.embedding import PositionalEncoding
from transformer.nn.feedforward import FeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer from "Attention Is All You Need".

    Each layer consists of:
    1. Multi-Head Self-Attention (with residual connection and layer norm)
    2. Position-wise Feed-Forward Network (with residual connection and layer norm)

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

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Self-attention weights
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attention_weights


class Encoder(nn.Module):
    """
    Transformer Encoder from "Attention Is All You Need".

    Consists of:
    1. Input Embedding
    2. Positional Encoding
    3. Stack of N Encoder Layers

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension (embedding size)
        n_layers: Number of encoder layers (default: 6)
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

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            mask: Optional attention mask

        Returns:
            output: Encoder output of shape (batch_size, seq_len, d_model)
            attention_weights: List of attention weights from each layer
        """
        # Embed tokens and scale by sqrt(d_model) as in the paper
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through encoder layers
        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_weights_list.append(attention_weights)

        # Final layer normalization
        x = self.norm(x)

        return x, attention_weights_list
