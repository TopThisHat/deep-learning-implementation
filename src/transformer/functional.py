"""Functional operations for Transformer models."""

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        query: Query tensor of shape (..., seq_len, d_k)
        key: Key tensor of shape (..., seq_len, d_k)
        value: Value tensor of shape (..., seq_len, d_v)
        mask: Optional mask tensor, where 0 indicates positions to mask out
        dropout_p: Dropout probability (default: 0.0)

    Returns:
        output: Attention output of shape (..., seq_len, d_v)
        attention_weights: Attention weights of shape (..., seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Compute attention scores: Q * K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


def create_look_ahead_mask(
    size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a look-ahead (causal) mask for autoregressive decoding.

    Prevents positions from attending to subsequent positions.
    Returns a lower triangular matrix of ones.

    Args:
        size: Sequence length
        device: Device to create the mask on (default: None, uses CPU)

    Returns:
        mask: Lower triangular mask of shape (1, 1, size, size)
              where 1 = attend, 0 = mask out
    """
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def create_padding_mask(
    seq: torch.Tensor,
    pad_token: int = 0,
) -> torch.Tensor:
    """
    Create a padding mask to ignore padding tokens.

    Args:
        seq: Input sequence of shape (batch_size, seq_len)
        pad_token: Token ID used for padding (default: 0)

    Returns:
        mask: Padding mask of shape (batch_size, 1, 1, seq_len)
              where 1 = attend, 0 = mask out
    """
    mask = (seq != pad_token).float()
    return mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
