"""Position-wise Feed-Forward Network module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network from "Attention Is All You Need".

    FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2

    Consists of two linear transformations with a ReLU activation in between.

    Args:
        d_model: Model dimension (embedding size)
        d_ff: Inner layer dimension (default: 2048, typically 4 * d_model)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
