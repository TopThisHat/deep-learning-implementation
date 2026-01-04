"""
Transformer: A PyTorch implementation of "Attention Is All You Need".

This library provides a clean, modular implementation of the Transformer
architecture as described in Vaswani et al. (2017).

Example:
    >>> import transformer
    >>> model = transformer.Transformer(
    ...     src_vocab_size=30000,
    ...     tgt_vocab_size=30000,
    ...     d_model=512,
    ... )
    >>> # Or use individual components
    >>> attention = transformer.nn.MultiHeadAttention(d_model=512, n_heads=8)
"""

from transformer.nn import (
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Generator,
)
from transformer.model import Transformer
from transformer.functional import (
    scaled_dot_product_attention,
    create_look_ahead_mask,
    create_padding_mask,
)
from transformer.optim import NoamScheduler, create_noam_optimizer
from transformer.loss import LabelSmoothingLoss

__version__ = "0.1.0"

__all__ = [
    # Model
    "Transformer",
    # Layers (nn)
    "MultiHeadAttention",
    "PositionalEncoding",
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "Encoder",
    "Decoder",
    "Generator",
    # Functional
    "scaled_dot_product_attention",
    "create_look_ahead_mask",
    "create_padding_mask",
    # Optimization
    "NoamScheduler",
    "create_noam_optimizer",
    # Loss
    "LabelSmoothingLoss",
]
