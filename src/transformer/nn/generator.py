"""Generator module for output projection."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator layer that projects decoder output to vocabulary logits.

    Typically the final layer of a Transformer model for language modeling.

    Args:
        d_model: Model dimension (embedding size)
        vocab_size: Size of the vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder output of shape (batch_size, seq_len, d_model)

        Returns:
            logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size)
        """
        return self.projection(x)
